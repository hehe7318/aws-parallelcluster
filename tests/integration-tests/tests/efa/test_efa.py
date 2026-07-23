# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file.
# This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied.
# See the License for the specific language governing permissions and limitations under the License.
import logging
import os
from datetime import datetime, timedelta, timezone

import boto3
import pytest
import xmltodict
from assertpy import assert_that, soft_assertions
from cfn_stacks_factory import CfnStack
from constants import REPOSITORY_ROOT
from remote_command_executor import RemoteCommandExecutor
from utils import generate_stack_name, get_compute_nodes_instance_ids, get_instance_info

from tests.common.assertions import assert_no_errors_in_logs
from tests.common.mpi_common import _test_mpi
from tests.common.nccl_common import install_and_run_nccl_benchmarks
from tests.common.utils import (
    fetch_instance_slots,
    get_capacity_reservation_id,
    read_remote_file,
    run_system_analyzer,
    wait_process_completion,
)

FSX_MOUNT_DIR = "/fsx"

# Default number of EFA interfaces the FSx-for-Lustre setup binds to LNet per instance type, mirroring
# the "Step 4: EFA interfaces" table in the FSx configure-efa-clients doc. Instances not listed fall back
# to the network-card rule (multiple network cards -> 2, single network card -> 1). This is the count the
# setup binds by design -- NOT the hardware EFA-device count (which can be far higher, e.g. 32 on p5).
# https://docs.aws.amazon.com/fsx/latest/LustreGuide/configure-efa-clients.html#add-efa-interfaces
FSX_EFA_BOUND_DEVICES_BY_INSTANCE_TYPE = {
    "p6-b300.48xlarge": 16,
    "p6e-gb200.36xlarge": 8,
    "p6-b200.48xlarge": 8,
    "p5en.48xlarge": 8,
    "p5e.48xlarge": 8,
    "p5.48xlarge": 8,
}

# Candidate head node instance types for when compute is p* or hpc* (one per family).
HEAD_NODE_CANDIDATES_X86 = [
    "c5.18xlarge",
    "c6i.16xlarge",
    "c7i.16xlarge",
    "m5.16xlarge",
    "m6i.16xlarge",
    "m7i.16xlarge",
    "r5.16xlarge",
]
HEAD_NODE_CANDIDATES_ARM = [
    "c6g.16xlarge",
    "c7g.16xlarge",
    "m6g.16xlarge",
    "m7g.16xlarge",
    "r6g.16xlarge",
]

FABTESTS_BASIC_TESTS = ["rdm_tagged_bw", "rdm_tagged_pingpong"]

FABTESTS_GDRCOPY_TESTS = ["runt"]


def _try_reserve_head_node_instance(region, az_id, architecture, os):
    """Try to create a 1-hour capacity reservation for a head node instance in the given AZ.

    Iterates through candidate instance types (one per family) and returns the first one that succeeds.
    Returns the selected instance type. Falls back to the first candidate if all reservations fail.
    """
    ec2_client = boto3.client("ec2", region_name=region)
    candidates = HEAD_NODE_CANDIDATES_X86 if architecture == "x86_64" else HEAD_NODE_CANDIDATES_ARM
    instance_platform = "Red Hat Enterprise Linux" if "rhel" in os else "Linux/UNIX"
    end_date = datetime.now(timezone.utc) + timedelta(hours=1)

    for candidate in candidates:
        try:
            response = ec2_client.create_capacity_reservation(
                InstanceType=candidate,
                InstancePlatform=instance_platform,
                AvailabilityZoneId=az_id,
                InstanceCount=1,
                EndDateType="limited",
                EndDate=end_date,
                Tenancy="default",
            )
            cr_id = response["CapacityReservation"]["CapacityReservationId"]
            logging.info("Created head node capacity reservation %s for %s in %s", cr_id, candidate, az_id)
            return candidate
        except Exception as e:
            logging.info("Capacity reservation for head node %s failed in %s: %s", candidate, az_id, e)

    # All candidates failed
    pytest.fail(
        "Could not reserve capacity for any head node instance type candidate",
    )
    return None


def _expected_efa_bound_devices(instance, region):
    """Return the number of EFA interfaces the FSx-for-Lustre setup is expected to bind on ``instance``.

    Mirrors the FSx configure-efa-clients "Step 4: EFA interfaces" table: a curated per-instance-type
    count for the high-EFA families, otherwise the network-card fallback (multiple network cards -> 2,
    single network card -> 1). EFA-for-Lustre is not p-family specific; any EFA-capable instance binds at
    least one interface, so this returns a value for every EFA instance.
    """
    curated = FSX_EFA_BOUND_DEVICES_BY_INSTANCE_TYPE.get(instance)
    if curated is not None:
        return curated
    network_cards = get_instance_info(instance, region)["NetworkInfo"]["NetworkCards"]
    return 2 if len(network_cards) > 1 else 1


def _provision_efa_fsx_stack(region, request, vpc_stack, cfn_stacks_factory):
    """Provision (or reuse) the external-shared-storage stack's EFA-enabled FSx Lustre + EFA security groups.

    Reuses the shared storage-stack.yaml CloudFormation template (the same one the update tests use)
    rather than creating FSx/SGs by hand: it enables only the EFA-FSx resources -- an EfaEnabled
    PERSISTENT_2 file system plus the two cross-referencing EFA security groups (client + file system,
    authorizing each other by SG-ID, since SRD is authorized by SG membership, not by CIDR) -- all placed
    in the compute-node subnet/AZ. When --external-shared-storage-stack-name is supplied, that pre-existing
    stack is reused instead of creating a new one. Cleanup is handled by cfn_stacks_factory. Returns
    (fsx_fs_id, client_security_group_id) from the stack outputs; the client SG rides the cluster and the
    file-system SG (cross-referencing it) rides the file system.
    """
    existing_stack_name = request.config.getoption("external_shared_storage_stack_name")
    if existing_stack_name:
        stack = CfnStack(name=existing_stack_name, region=region, template=None)
    else:
        # The FSx file system must live in the same AZ (subnet) as the compute nodes for EFA to work.
        compute_subnet_id = vpc_stack.get_private_subnet()
        compute_subnet_az = boto3.resource("ec2", region_name=region).Subnet(compute_subnet_id).availability_zone
        params = [
            {"ParameterKey": "Vpc", "ParameterValue": vpc_stack.cfn_outputs["VpcId"]},
            {"ParameterKey": "SubnetOne", "ParameterValue": compute_subnet_id},
            # EbsVolumeAz is a required parameter even though we create no EBS; pass the compute subnet's AZ.
            {"ParameterKey": "EbsVolumeAz", "ParameterValue": compute_subnet_az},
            # Enable ONLY the EFA-FSx resources; everything else in the storage stack is off.
            {"ParameterKey": "CreateEbs", "ParameterValue": "false"},
            {"ParameterKey": "CreateEfs", "ParameterValue": "false"},
            {"ParameterKey": "CreateFsxLustre", "ParameterValue": "false"},
            {"ParameterKey": "CreateEfaFsxLustre", "ParameterValue": "true"},
            {"ParameterKey": "CreateFsxOntap", "ParameterValue": "false"},
            {"ParameterKey": "CreateFsxOpenZfs", "ParameterValue": "false"},
            {"ParameterKey": "CreateFileCache", "ParameterValue": "false"},
        ]
        template_path = os.path.join(REPOSITORY_ROOT, "cloudformation/storage/storage-stack.yaml")
        with open(template_path, encoding="utf-8") as template_file:
            template = template_file.read()
        stack = CfnStack(
            name=generate_stack_name("integ-tests-efa-fsx-storage", request.config.getoption("stackname_suffix")),
            region=region,
            parameters=params,
            template=template,
            capabilities=["CAPABILITY_IAM"],
        )
        cfn_stacks_factory.create_stack(stack)
    return stack.cfn_outputs["EfaFsxLustreFsId"], stack.cfn_outputs["EfaFsxLustreClientSecurityGroupId"]


@pytest.mark.usefixtures("flags")
def test_efa(
    os,
    region,
    scheduler,
    instance,
    pcluster_config_reader,
    clusters_factory,
    test_datadir,
    architecture,
    scheduler_commands_factory,
    request,
    vpc_stack,
    cfn_stacks_factory,
):
    """
    Test all EFA Features.

    Grouped all tests in a single function so that cluster can be reused for all of them.
    """
    head_node_instance = instance
    if len(get_instance_info(instance, region)["NetworkInfo"]["NetworkCards"]) > 1:
        az_id = vpc_stack.az_override or vpc_stack.default_az_id
        head_node_instance = _try_reserve_head_node_instance(region, az_id, architecture, os)
    max_queue_size = 2
    capacity_reservation_id = None
    # p5 and p6 family instances need capacity blocks, and so placement group is set to false.
    capacity_block_instance_type = instance.startswith("p5") or instance.startswith("p6")
    placement_group_enabled = not capacity_block_instance_type
    if capacity_block_instance_type:
        capacity_reservations_ids = get_capacity_reservation_id(request, instance, region, max_queue_size, os)
        if capacity_reservations_ids:
            capacity_reservation_id = capacity_reservations_ids[0].get("CapacityReservationId")
        else:
            message = f"Skipping the test as no Capacity Block for {instance} and os {os} was found in {region}"
            logging.warn(message)
            pytest.skip(message)

    # Provision the EFA-enabled FSx Lustre file system and its two cross-referencing EFA security groups
    # (client + file system) via the shared external-shared-storage CloudFormation stack. EFA for Lustre
    # works on any EFA-capable instance (the setup binds a per-instance-type number of EFA interfaces --
    # see _expected_efa_bound_devices), so this runs for every EFA instance, not just p*. The cluster
    # attaches the client SG, which cross-references the file-system SG, so EFA's SRD path (authorized by
    # SG membership) is permitted between compute nodes and the file system. The OnNodeStart script (staged
    # in S3) runs the official FSx EFA-Lustre client setup on each node.
    fsx_fs_id, efa_fsx_security_group_id = _provision_efa_fsx_stack(region, request, vpc_stack, cfn_stacks_factory)

    bucket_name = request.getfixturevalue("s3_bucket_factory")()
    boto3.resource("s3", region_name=region).Bucket(bucket_name).upload_file(
        str(test_datadir / "pcluster-efa-fsx-lustre-client-tutorial.sh"),
        "pcluster-efa-fsx-lustre-client-tutorial.sh",
    )

    slots_per_instance = fetch_instance_slots(region, instance, multithreading_disabled=True)
    cluster_config = pcluster_config_reader(
        head_node_instance=head_node_instance,
        max_queue_size=max_queue_size,
        capacity_reservation_id=capacity_reservation_id,
        placement_group_enabled=placement_group_enabled,
        fsx_mount_dir=FSX_MOUNT_DIR,
        fsx_fs_id=fsx_fs_id,
        fsx_efa_security_group_id=efa_fsx_security_group_id,
        bucket_name=bucket_name,
    )
    cluster = clusters_factory(cluster_config)
    remote_command_executor = RemoteCommandExecutor(cluster)
    scheduler_commands = scheduler_commands_factory(remote_command_executor)

    _test_efa_installation(scheduler_commands, remote_command_executor, efa_installed=True, partition="efa-enabled")
    _test_efa_eni_configuration(cluster, region)
    _test_mpi(remote_command_executor, slots_per_instance, scheduler, scheduler_commands, partition="efa-enabled")
    logging.info("Running on Instances: {0}".format(get_compute_nodes_instance_ids(cluster.cfn_name, region)))

    run_system_analyzer(cluster, scheduler_commands_factory, request, partition="efa-enabled")

    _test_shm_transfer_is_enabled(scheduler_commands, remote_command_executor, partition="efa-enabled")

    if instance.startswith("p"):
        # Doc of supported instance types and operating systems:
        # https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start-nccl.html
        install_and_run_nccl_benchmarks(remote_command_executor, "openmpi", scheduler_commands, instance, os)

    with soft_assertions():
        assert_no_errors_in_logs(remote_command_executor, scheduler, skip_ice=True)
    if "us-iso" not in region:
        # Run Fabric tests. Fabric tests require Internet connection, so cannot be run in us-iso regions
        run_system_analyzer(cluster, scheduler_commands_factory, request, partition="efa-enabled")

        fabtests_report = _execute_fabtests(remote_command_executor, test_datadir, instance)

        num_tests = int(fabtests_report.get("testsuites", {}).get("testsuite", {})[0].get("@tests", None))
        num_failures = int(fabtests_report.get("testsuites", {}).get("testsuite", {})[0].get("@failures", None))
        num_errors = int(fabtests_report.get("testsuites", {}).get("testsuite", {})[0].get("@errors", None))

        with soft_assertions():
            assert_that(num_tests, description="Cannot read number of tests from Fabtests report").is_not_none()
            assert_that(num_failures, description="Cannot read number of failures from Fabtests report").is_not_none()
            assert_that(num_errors, description="Cannot read number of errors from Fabtests report").is_not_none()

        if num_failures + num_errors > 0:
            logging.info(f"Fabtests report:\n{fabtests_report}")

        with soft_assertions():
            assert_that(
                num_failures, description=f"{num_failures}/{num_tests} libfabric tests are failing"
            ).is_equal_to(0)
            assert_that(num_errors, description=f"{num_errors}/{num_tests} libfabric tests got errors").is_equal_to(0)
            assert_no_errors_in_logs(remote_command_executor, scheduler, skip_ice=True)

    # EFA-for-Lustre validation (folded in from the standalone test_efa_enabled_fsx_lustre): the client must
    # bind the expected number of EFA devices, the @efa data path to the file system must work, the OSC (data)
    # targets must ride @efa, and a basic read/write must succeed.
    with soft_assertions():
        _test_efa_fsx_device_count(
            scheduler_commands, remote_command_executor, region, instance, partition="efa-enabled"
        )
        _test_lnet_efa_ping(scheduler_commands, remote_command_executor, region, fsx_fs_id, partition="efa-enabled")
        _test_lustre_targets_use_efa(scheduler_commands, remote_command_executor, partition="efa-enabled")
        _test_fsx_read_write(scheduler_commands, remote_command_executor, FSX_MOUNT_DIR, partition="efa-enabled")


def _test_efa_eni_configuration(cluster, region):
    """Verify compute nodes have the expected EFA network interface configuration.

    Under the new default (PC 3.15):
    - Each compute node should have exactly one private IP address (on the interface ENI at NCI-0)
    - All ENIs except one should be efa-only (no IP, EFA fabric only)
    """
    ec2_client = boto3.client("ec2", region_name=region)
    compute_instance_ids = get_compute_nodes_instance_ids(cluster.cfn_name, region)

    for instance_id in compute_instance_ids:
        instance_info = ec2_client.describe_instances(InstanceIds=[instance_id])["Reservations"][0]["Instances"][0]
        enis = instance_info["NetworkInterfaces"]
        logging.info(f"Instance {instance_id} has {len(enis)} ENIs")

        enis_with_ip = []
        efa_only_enis = []
        for eni in enis:
            interface_type = eni.get("InterfaceType", "interface")
            private_ips = eni.get("PrivateIpAddresses", [])
            attachment = eni.get("Attachment", {})
            network_card_index = attachment.get("NetworkCardIndex", 0)
            device_index = attachment.get("DeviceIndex", 0)
            logging.info(
                f"  ENI {eni['NetworkInterfaceId']}: InterfaceType={interface_type}, "
                f"PrivateIPs={len(private_ips)}, NetworkCardIndex={network_card_index}, DeviceIndex={device_index}"
            )
            if interface_type == "efa-only":
                efa_only_enis.append(eni)
            else:
                enis_with_ip.append(eni)

        # Exactly one ENI should have a private IP (the interface ENI on NCI-0)
        assert_that(
            len(enis_with_ip), description=f"Instance {instance_id} should have exactly 1 ENI with a private IP"
        ).is_equal_to(1)

        # All other ENIs should be efa-only
        if len(enis) > 1:
            assert_that(
                len(efa_only_enis), description=f"Instance {instance_id}: all ENIs except one should be efa-only"
            ).is_equal_to(len(enis) - 1)

        logging.info(
            f"Instance {instance_id}: {len(enis_with_ip)} ENI(s) with private IP, {len(efa_only_enis)} efa-only ENI(s)"
        )


def _execute_fabtests(remote_command_executor, test_datadir, instance):
    fabtests_dir = "/shared/fabtests"
    fabtests_pid_file = f"{fabtests_dir}/outputs/fabtests.pid"
    fabtests_log_file = f"{fabtests_dir}/outputs/fabtests.log"
    fabtests_report_file = f"{fabtests_dir}/outputs/fabtests.report"

    logging.info("Installing Fabtests")
    remote_command_executor.run_remote_script(
        str(test_datadir / "install-fabtests.sh"), args=[fabtests_dir], timeout=600
    )

    logging.info("Running Fabtests")
    test_cases = FABTESTS_BASIC_TESTS + FABTESTS_GDRCOPY_TESTS if instance == "p4d.24xlarge" else FABTESTS_BASIC_TESTS

    if "g6" in instance:
        test_cases = test_cases + ["not cuda"]

    remote_command_executor.run_remote_script(
        str(test_datadir / "run-fabtests.sh"),
        args=[
            fabtests_dir,
            fabtests_pid_file,
            fabtests_log_file,
            fabtests_report_file,
            "efa-enabled-st-efa-enabled-i1-1",
            "efa-enabled-st-efa-enabled-i1-2",
            ",".join(test_cases),
            "enable-gdr" if instance == "p4d.24xlarge" else "skip-gdr",
        ],
        timeout=60,
        pty=False,
    )

    pid = read_remote_file(remote_command_executor, fabtests_pid_file)

    wait_process_completion(remote_command_executor, pid)

    logging.info("Retrieving Fabtests report")
    report_content = read_remote_file(remote_command_executor, fabtests_report_file)
    logging.info("Parsing Fabtests report")
    return xmltodict.parse(report_content)


def _test_efa_installation(scheduler_commands, remote_command_executor, efa_installed=True, partition=None):
    # Output contains:
    # 00:06.0 Ethernet controller: Amazon.com, Inc. Device efa0
    logging.info("Testing EFA installed")
    if partition:
        result = scheduler_commands.submit_command("lspci -n > /shared/lspci.out", partition=partition)
    else:
        result = scheduler_commands.submit_command("lspci -n > /shared/lspci.out")

    job_id = scheduler_commands.assert_job_submitted(result.stdout)
    scheduler_commands.wait_job_completed(job_id)
    scheduler_commands.assert_job_succeeded(job_id)

    # Check if EFA interface is on compute node
    result = remote_command_executor.run_remote_command("cat /shared/lspci.out")
    if efa_installed:
        assert_that(result.stdout).contains("1d0f:efa")
    else:
        assert_that(result.stdout).does_not_contain("1d0f:efa")

    # Check EFA interface not present on head node
    result = remote_command_executor.run_remote_command("lspci -n")
    assert_that(result.stdout).does_not_contain("1d0f:efa")


def _test_shm_transfer_is_enabled(scheduler_commands, remote_command_executor, partition=None):
    logging.info("Testing SHM Transfer is enabled")
    if partition:
        result = scheduler_commands.submit_command("fi_info -p efa 2>&1 > /shared/fi_info.out", partition=partition)
    else:
        result = scheduler_commands.submit_command("fi_info -p efa 2>&1 > /shared/fi_info.out")
    job_id = scheduler_commands.assert_job_submitted(result.stdout)
    scheduler_commands.wait_job_completed(job_id)
    job_stdout = remote_command_executor.run_remote_command(f"cat slurm-{job_id}.out").stdout
    logging.info(f"Job stdout is: {job_stdout}")
    scheduler_commands.assert_job_succeeded(job_id)
    result = remote_command_executor.run_remote_command("cat /shared/fi_info.out")
    assert_that(result.stdout).does_not_contain("SHM transfer will be disabled because of ptrace protection")


def _submit_and_get_output(scheduler_commands, remote_command_executor, command, partition):
    """Submit a command on an EFA compute node, wait for it, and return the job's captured stdout.

    The command runs as a Slurm job on ``partition`` (the EFA-enabled compute queue), so all EFA/Lustre
    probing happens on a compute node, not the head node. The job's stdout is written by Slurm to
    ``slurm-<job_id>.out`` in the submitting user's home (shared with the head node), which we then read
    from the head node.
    """
    result = scheduler_commands.submit_command(command, partition=partition)
    job_id = scheduler_commands.assert_job_submitted(result.stdout)
    scheduler_commands.wait_job_completed(job_id)
    scheduler_commands.assert_job_succeeded(job_id)
    return remote_command_executor.run_remote_command(f"cat slurm-{job_id}.out").stdout


def _test_efa_fsx_device_count(scheduler_commands, remote_command_executor, region, instance, partition=None):
    """Assert LNet binds the expected number of EFA devices (regression guard for the under-bind bug).

    The expected count is the FSx doc's per-instance-type default (see FSX_EFA_BOUND_DEVICES_BY_INSTANCE_TYPE
    and the network-card fallback) -- the number the setup binds BY DESIGN, not the hardware EFA-device count
    (``/sys/class/infiniband`` can expose far more, e.g. 32 on p5.48xlarge). If fewer than expected are bound,
    Lustre silently falls back to TCP on the unbound devices.
    """
    logging.info("Testing that LNet binds the expected number of EFA devices for FSx")

    expected_efa_devices = _expected_efa_bound_devices(instance, region)
    logging.info("Instance type %s is expected to bind %s EFA interface(s) for FSx", instance, expected_efa_devices)

    lnet_out = _submit_and_get_output(
        scheduler_commands,
        remote_command_executor,
        "sudo lnetctl net show --net efa",
        partition,
    )
    logging.info("lnetctl net show --net efa output:\n%s", lnet_out)
    # Each bound EFA interface appears as an "interfaces:" entry in the lnetctl output.
    bound_efa_interfaces = lnet_out.count("interfaces:")
    assert_that(
        bound_efa_interfaces,
        description=(
            f"LNet bound {bound_efa_interfaces} EFA interface(s) but {instance} is expected to bind "
            f"{expected_efa_devices}; Lustre would fall back to TCP on the unbound devices"
        ),
    ).is_equal_to(expected_efa_devices)


def _test_lnet_efa_ping(scheduler_commands, remote_command_executor, region, fsx_fs_id, partition=None):
    """Assert an LNet ping over @efa to the file-system host succeeds."""
    logging.info("Testing lnetctl ping over @efa to the FSx file system")

    # Resolve the private IP of the FSx file system to build its LNet NID.
    fsx = boto3.client("fsx", region_name=region).describe_file_systems(FileSystemIds=[fsx_fs_id])["FileSystems"][0]
    fs_dns_name = fsx["DNSName"]
    fs_ip = ""
    network_interface_ids = fsx.get("NetworkInterfaceIds", [])
    if network_interface_ids:
        eni = boto3.client("ec2", region_name=region).describe_network_interfaces(
            NetworkInterfaceIds=[network_interface_ids[0]]
        )["NetworkInterfaces"][0]
        fs_ip = eni.get("PrivateIpAddress", "")
    fs_host = fs_ip or fs_dns_name
    logging.info("FSx file system LNet host resolved to %s (dns=%s)", fs_host, fs_dns_name)

    # NOTE: submit_command wraps this in sbatch --wrap='...', so the command must contain no single quotes
    # (a nested single quote would close the wrap early). Extract the NID with a double-quoted sed.
    ping_out = _submit_and_get_output(
        scheduler_commands,
        remote_command_executor,
        (
            'local_nid=$(sudo lnetctl net show --net efa | grep -m1 nid: | sed "s/.*nid: //"); '
            f"sudo lnetctl ping --source ${{local_nid}} {fs_host}@efa"
        ),
        partition,
    )
    logging.info("lnetctl ping over @efa output:\n%s", ping_out)
    assert_that(ping_out).does_not_contain("failed")
    assert_that(ping_out).contains("@efa")


def _test_lustre_targets_use_efa(scheduler_commands, remote_command_executor, partition=None):
    """Assert the Lustre OSC (data) targets are connected over @efa.

    Only the OSC/OST (data) imports are expected to ride @efa. The MDC/MDT (metadata) import stays on @tcp
    by design when the file system uses AUTOMATIC metadata: the FSx-managed metadata server sits on a plain
    (non-EFA) NIC, so its target is served over @tcp and must NOT be asserted to be @efa.
    """
    logging.info("Testing that Lustre OSC (data) targets are connected over @efa")

    osc_out = _submit_and_get_output(
        scheduler_commands,
        remote_command_executor,
        "sudo lctl get_param 'osc.*.import'",
        partition,
    )
    logging.info("Lustre osc import output:\n%s", osc_out)
    # The OSC (data) connection must ride the efa LNet.
    assert_that(osc_out).contains("@efa")
    assert_that(osc_out).does_not_contain("@tcp")


def _test_fsx_read_write(scheduler_commands, remote_command_executor, mount_dir, partition=None):
    """Assert a basic write/read round-trip to the mounted FSx file system succeeds."""
    logging.info("Testing basic write/read to the mounted FSx file system")

    marker = "hello-efa-fsx-lustre"
    test_file = f"{mount_dir}/efa_fsx_lustre_test.txt"
    rw_out = _submit_and_get_output(
        scheduler_commands,
        remote_command_executor,
        f"echo '{marker}' > {test_file} && sync && cat {test_file}",
        partition,
    )
    logging.info("FSx write/read output:\n%s", rw_out)
    assert_that(rw_out).contains(marker)
