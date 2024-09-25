#!/usr/bin/env python3

# -------------------------------------------------------
# Script: auto_update.py
#
# Description:
# This script allows management of automatic updates on supported Linux distributions (Ubuntu, Debian, Red Hat, Fedora).
# It can enable or disable automatic updates, configure update settings, and manually install updates.
# It supports installing all updates, security updates only, checking for available updates, excluding specific packages,
# and scheduling updates at specific times. It provides options to reboot the system if necessary after updates,
# simulate updates (dry run), and log actions. Additionally, it can clean up package caches and remove unused packages.
#
# Usage:
# ./auto_update.py [options]
#
# Options:
# -i, --install             Install all available updates.
# -c, --cleanup             Clean up package cache and remove unused packages.
# -s, --security-only       Install security updates only.
# -e, --enable-auto-update  Enable automatic updates.
# -d, --disable-auto-update Disable automatic updates.
# -U, --check-updates       Check for available updates.
# -l, --list-updates        List available updates.
# -x, --exclude PACKAGES    Exclude specified packages from updates (comma-separated list).
# -t, --schedule TIME       Schedule updates at specified time (HH:MM).
# -r, --reboot              Reboot system after updates if necessary.
# -n, --dry-run             Simulate updates without installing.
# -v, --verbose             Enable verbose output (INFO level).
# -vv, --debug              Enable debug output (DEBUG level).
# -h, --help                Show help message and exit.
# -o, --log-file FILE       Log actions to specified file.
#
# Returns:
# Exit code 0 on success, non-zero on failure.
#
# Requirements:
# - Only default Python3 modules.
#
# -------------------------------------------------------
# © 2024 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional, Tuple


def parse_arguments() -> Tuple[argparse.Namespace, argparse.ArgumentParser]:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Manage automatic updates on supported Linux distributions.'
    )
    parser.add_argument(
        '-i', '--install',
        action='store_true',
        help='Install all available updates.'
    )
    parser.add_argument(
        '-c', '--cleanup',
        action='store_true',
        help='Clean up package cache and remove unused packages.'
    )
    parser.add_argument(
        '-s', '--security-only',
        action='store_true',
        help='Install security updates only.'
    )
    parser.add_argument(
        '-e', '--enable-auto-update',
        action='store_true',
        help='Enable automatic updates.'
    )
    parser.add_argument(
        '-d', '--disable-auto-update',
        action='store_true',
        help='Disable automatic updates.'
    )
    parser.add_argument(
        '-U', '--check-updates',
        action='store_true',
        help='Check for available updates.'
    )
    parser.add_argument(
        '-l', '--list-updates',
        action='store_true',
        help='List available updates.'
    )
    parser.add_argument(
        '-x', '--exclude',
        type=str,
        help='Exclude specified packages from updates (comma-separated list).'
    )
    parser.add_argument(
        '-t', '--schedule',
        type=str,
        help='Schedule updates at specified time (HH:MM).'
    )
    parser.add_argument(
        '-r', '--reboot',
        action='store_true',
        help='Reboot system after updates if necessary.'
    )
    parser.add_argument(
        '-n', '--dry-run',
        action='store_true',
        help='Simulate updates without installing.'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output (INFO level).'
    )
    parser.add_argument(
        '-vv', '--debug',
        action='store_true',
        help='Enable debug output (DEBUG level).'
    )
    parser.add_argument(
        '-o', '--log-file',
        type=str,
        help='Log actions to specified file.'
    )
    args = parser.parse_args()
    return args, parser


def setup_logging(verbose: bool = False, debug: bool = False, log_file: Optional[str] = None) -> None:
    """
    Sets up the logging configuration.
    """
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.ERROR

    if log_file:
        logging.basicConfig(filename=log_file, level=level, format='%(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=level, format='%(levelname)s: %(message)s')


def get_distribution() -> str:
    """
    Detects the Linux distribution.
    """
    try:
        with open('/etc/os-release', 'r') as f:
            lines = f.readlines()
        distro_info = {}
        for line in lines:
            if '=' in line:
                key, value = line.rstrip().split('=', 1)
                value = value.strip('"')
                distro_info[key] = value
        distro_id = distro_info.get('ID', '').lower()
        return distro_id
    except Exception as e:
        logging.error(f"Could not determine Linux distribution: {e}")
        sys.exit(1)


class AutoUpdateManager:
    def __init__(self, dry_run: bool = False, exclude: Optional[List[str]] = None, use_sudo: bool = False):
        self.dry_run = dry_run
        self.exclude = exclude if exclude else []
        self.use_sudo = use_sudo


    def install_updates(self, security_only: bool = False):
        """
        Install updates.
        """
        raise NotImplementedError


    def enable_auto_updates(self):
        """
        Enable automatic updates.
        """
        raise NotImplementedError


    def disable_auto_updates(self):
        """
        Disable automatic updates.
        """
        raise NotImplementedError


    def check_updates(self):
        """
        Check for available updates.
        """
        raise NotImplementedError


    def list_updates(self):
        """
        List available updates.
        """
        raise NotImplementedError


    def schedule_updates(self, time_str: str):
        """
        Schedule updates at specified time.
        """
        raise NotImplementedError


    def reboot_if_needed(self):
        """
        Reboot the system if needed.
        """
        raise NotImplementedError


    def get_reboot_required(self) -> bool:
        """
        Check if a reboot is required.
        """
        raise NotImplementedError


    def cleanup(self):
        """
        Clean up package cache and remove unused packages.
        """
        raise NotImplementedError


class APTManager(AutoUpdateManager):
    def install_updates(self, security_only: bool = False):
        """
        Install updates using apt.
        """
        cmd_update = ['apt-get', 'update']
        cmd_upgrade = ['apt-get', 'upgrade', '-y']
        if security_only:
            # For security updates, typically use unattended-upgrades or specify security repositories
            # This is a placeholder; actual implementation may vary
            cmd_upgrade = ['apt-get', 'upgrade', '-y', '-o', 'Dir::Etc::sourcelist=/etc/apt/security.sources.list']
        if self.dry_run:
            cmd_upgrade.append('--dry-run')
        if self.exclude:
            logging.warning('Excluding packages is not directly supported with apt-get upgrade.')
            # Alternatively, mark packages as held
            held_packages = ' '.join(self.exclude)
            hold_cmd = ['apt-mark', 'hold'] + self.exclude
            unhold_cmd = ['apt-mark', 'unhold'] + self.exclude
            if self.use_sudo:
                hold_cmd = ['sudo'] + hold_cmd
                unhold_cmd = ['sudo'] + unhold_cmd
            subprocess.run(hold_cmd, check=True)
            logging.info(f"Excluded packages from updates: {held_packages}")

        if self.use_sudo:
            cmd_update = ['sudo'] + cmd_update
            cmd_upgrade = ['sudo'] + cmd_upgrade

        logging.info('Updating package lists...')
        try:
            subprocess.run(cmd_update, check=True)
            logging.info('Installing updates...')
            subprocess.run(cmd_upgrade, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error installing updates: {e}")
            sys.exit(1)
        finally:
            if self.exclude:
                # Unhold the packages after update
                subprocess.run(unhold_cmd, check=True)
                logging.info(f"Removed hold on packages: {', '.join(self.exclude)}")


    def enable_auto_updates(self):
        """
        Enable automatic updates.
        """
        auto_upgrade_file = '/etc/apt/apt.conf.d/20auto-upgrades'
        content = 'APT::Periodic::Update-Package-Lists "1";\nAPT::Periodic::Unattended-Upgrade "1";\n'
        if self.use_sudo:
            cmd = ['sudo', 'tee', auto_upgrade_file]
            try:
                process = subprocess.run(cmd, input=content.encode(), check=True)
                logging.info('Automatic updates enabled.')
            except subprocess.CalledProcessError as e:
                logging.error(f'Failed to enable automatic updates: {e}')
                sys.exit(1)
        else:
            try:
                with open(auto_upgrade_file, 'w') as f:
                    f.write(content)
                logging.info('Automatic updates enabled.')
            except Exception as e:
                logging.error(f'Failed to enable automatic updates: {e}')
                sys.exit(1)


    def disable_auto_updates(self):
        """
        Disable automatic updates.
        """
        auto_upgrade_file = '/etc/apt/apt.conf.d/20auto-upgrades'
        content = 'APT::Periodic::Update-Package-Lists "0";\nAPT::Periodic::Unattended-Upgrade "0";\n'
        if self.use_sudo:
            cmd = ['sudo', 'tee', auto_upgrade_file]
            try:
                process = subprocess.run(cmd, input=content.encode(), check=True)
                logging.info('Automatic updates disabled.')
            except subprocess.CalledProcessError as e:
                logging.error(f'Failed to disable automatic updates: {e}')
                sys.exit(1)
        else:
            try:
                with open(auto_upgrade_file, 'w') as f:
                    f.write(content)
                logging.info('Automatic updates disabled.')
            except Exception as e:
                logging.error(f'Failed to disable automatic updates: {e}')
                sys.exit(1)


    def check_updates(self):
        """
        Check for available updates.
        """
        cmd = ['apt-get', 'update']
        if self.use_sudo:
            cmd = ['sudo'] + cmd
        logging.info('Updating package lists...')
        try:
            subprocess.run(cmd, check=True)
            logging.info('Package lists updated successfully.')
        except subprocess.CalledProcessError as e:
            logging.error(f"Error checking updates: {e}")
            sys.exit(1)


    def list_updates(self):
        """
        List available updates.
        """
        cmd = ['apt', 'list', '--upgradeable']
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error listing updates: {e}")
            sys.exit(1)


    def schedule_updates(self, time_str: str):
        """
        Schedule updates at specified time.
        """
        logging.error('Scheduling updates is not implemented.')
        sys.exit(1)


    def reboot_if_needed(self):
        """
        Reboot the system if needed.
        """
        if self.get_reboot_required():
            logging.info('Reboot is required. Rebooting now...')
            cmd = ['reboot']
            if self.use_sudo:
                cmd = ['sudo'] + cmd
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                logging.error(f"Error rebooting system: {e}")
                sys.exit(1)
        else:
            logging.info('No reboot is required.')


    def get_reboot_required(self) -> bool:
        """
        Check if a reboot is required.
        """
        return os.path.exists('/var/run/reboot-required')


    def cleanup(self):
        """
        Clean up package cache and remove unused packages using apt.
        """
        cmds = [
            ['apt-get', 'autoclean', '-y'],
            ['apt-get', 'autoremove', '-y']
        ]
        for cmd in cmds:
            if self.use_sudo:
                cmd = ['sudo'] + cmd
            logging.info(f'Running {" ".join(cmd)}')
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                logging.error(f"Error during cleanup: {e}")
                sys.exit(1)


class YUMManager(AutoUpdateManager):
    def install_updates(self, security_only: bool = False):
        """
        Install updates using yum or dnf.
        """
        # Determine whether to use yum or dnf
        pkg_manager = 'dnf' if Path('/usr/bin/dnf').exists() else 'yum'

        if self.dry_run:
            cmd_upgrade = [pkg_manager, 'upgrade', '-y', '--downloadonly']
        else:
            cmd_upgrade = [pkg_manager, 'upgrade', '-y']
        if security_only:
            # For security updates, use the appropriate plugin or repository
            # This is a placeholder; actual implementation may vary
            cmd_upgrade.extend(['--security'])
        if self.exclude:
            exclude_pkgs = ','.join(self.exclude)
            cmd_upgrade.append(f'--exclude={exclude_pkgs}')
            logging.info(f"Excluding packages from updates: {exclude_pkgs}")

        if self.use_sudo:
            cmd_upgrade = ['sudo'] + cmd_upgrade

        logging.info('Installing updates...')
        try:
            subprocess.run(cmd_upgrade, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error installing updates: {e}")
            sys.exit(1)


    def enable_auto_updates(self):
        """
        Enable automatic updates.
        """
        # Depending on the system, enable yum-cron or dnf-automatic
        if Path('/usr/bin/dnf').exists():
            # Fedora or newer RHEL/CentOS versions
            service = 'dnf-automatic.timer'
        else:
            # Older RHEL/CentOS versions
            service = 'yum-cron'
        cmd = ['systemctl', 'enable', '--now', service]
        if self.use_sudo:
            cmd = ['sudo'] + cmd
        try:
            subprocess.run(cmd, check=True)
            logging.info(f'Automatic updates enabled using {service}.')
        except subprocess.CalledProcessError as e:
            logging.error(f'Failed to enable automatic updates: {e}')
            sys.exit(1)


    def disable_auto_updates(self):
        """
        Disable automatic updates.
        """
        if Path('/usr/bin/dnf').exists():
            service = 'dnf-automatic.timer'
        else:
            service = 'yum-cron'
        cmd = ['systemctl', 'disable', '--now', service]
        if self.use_sudo:
            cmd = ['sudo'] + cmd
        try:
            subprocess.run(cmd, check=True)
            logging.info(f'Automatic updates disabled for {service}.')
        except subprocess.CalledProcessError as e:
            logging.error(f'Failed to disable automatic updates: {e}')
            sys.exit(1)


    def check_updates(self):
        """
        Check for available updates.
        """
        pkg_manager = 'dnf' if Path('/usr/bin/dnf').exists() else 'yum'
        cmd = [pkg_manager, 'check-update']
        if self.use_sudo:
            cmd = ['sudo'] + cmd
        logging.info('Checking for available updates...')
        try:
            subprocess.run(cmd, check=True)
            logging.info('Check-update completed successfully.')
        except subprocess.CalledProcessError as e:
            if e.returncode == 100:
                logging.info('Updates are available.')
            elif e.returncode == 0:
                logging.info('System is up to date.')
            else:
                logging.error(f"Error checking updates: {e}")
                sys.exit(1)


    def list_updates(self):
        """
        List available updates.
        """
        pkg_manager = 'dnf' if Path('/usr/bin/dnf').exists() else 'yum'
        cmd = [pkg_manager, 'list', 'updates']
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error listing updates: {e}")
            sys.exit(1)


    def schedule_updates(self, time_str: str):
        """
        Schedule updates at specified time.
        """
        logging.error('Scheduling updates is not implemented.')
        sys.exit(1)


    def reboot_if_needed(self):
        """
        Reboot the system if needed.
        """
        if self.get_reboot_required():
            logging.info('Reboot is required. Rebooting now...')
            cmd = ['reboot']
            if self.use_sudo:
                cmd = ['sudo'] + cmd
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                logging.error(f"Error rebooting system: {e}")
                sys.exit(1)
        else:
            logging.info('No reboot is required.')


    def get_reboot_required(self) -> bool:
        """
        Check if a reboot is required.
        """
        # For example, check if /var/run/reboot-required exists (common in Debian-based systems)
        # For Red Hat-based, it might require different checks
        return os.path.exists('/var/run/reboot-required')


    def cleanup(self):
        """
        Clean up package cache and remove unused packages using yum/dnf.
        """
        pkg_manager = 'dnf' if Path('/usr/bin/dnf').exists() else 'yum'
        cmds = [
            [pkg_manager, 'autoremove', '-y'],
            [pkg_manager, 'clean', 'all']
        ]
        for cmd in cmds:
            if self.use_sudo:
                cmd = ['sudo'] + cmd
            logging.info(f'Running {" ".join(cmd)}')
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                logging.error(f"Error during cleanup: {e}")
                sys.exit(1)


def main():
    """
    Main function to orchestrate the auto-update management process.
    """
    args, parser = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug, log_file=args.log_file)
    logging.debug('Parsed command-line arguments.')
    logging.info('Starting auto_update script.')

    # Check for root privileges
    if os.geteuid() != 0:
        # Not running as root
        # Check if sudo is available
        if shutil.which('sudo') is not None:
            use_sudo = True
            logging.info('Not running as root. Will use sudo for privileged commands.')
        else:
            logging.error('This script must be run as root or sudo must be available.')
            sys.exit(1)
    else:
        use_sudo = False

    distro_id = get_distribution()
    logging.debug(f'Detected Linux distribution: {distro_id}')
    manager = None
    if distro_id in ['ubuntu', 'debian']:
        exclude_list = args.exclude.split(',') if args.exclude else None
        manager = APTManager(dry_run=args.dry_run, exclude=exclude_list, use_sudo=use_sudo)
    elif distro_id in ['rhel', 'centos', 'redhat', 'fedora']:
        exclude_list = args.exclude.split(',') if args.exclude else None
        manager = YUMManager(dry_run=args.dry_run, exclude=exclude_list, use_sudo=use_sudo)
    else:
        logging.error(f'Distribution "{distro_id}" is not supported.')
        sys.exit(1)

    # If no action is specified, print help
    if not any([
        args.install,
        args.security_only,
        args.enable_auto_update,
        args.disable_auto_update,
        args.check_updates,
        args.list_updates,
        args.schedule,
        args.cleanup
    ]):
        parser.print_help()
        sys.exit(0)

    # Perform actions based on arguments
    if args.check_updates:
        logging.debug('Action: Check for updates.')
        manager.check_updates()
    if args.list_updates:
        logging.debug('Action: List available updates.')
        manager.list_updates()
    if args.install or args.security_only:
        logging.debug('Action: Install updates.')
        manager.install_updates(security_only=args.security_only)
    if args.enable_auto_update:
        logging.debug('Action: Enable automatic updates.')
        manager.enable_auto_updates()
    if args.disable_auto_update:
        logging.debug('Action: Disable automatic updates.')
        manager.disable_auto_updates()
    if args.schedule:
        logging.debug(f'Action: Schedule updates at {args.schedule}.')
        manager.schedule_updates(args.schedule)
    if args.cleanup:
        logging.debug('Action: Cleanup package cache and remove unused packages.')
        manager.cleanup()
    if args.reboot:
        logging.debug('Action: Reboot if needed.')
        manager.reboot_if_needed()

    logging.info('auto_update script completed successfully.')


if __name__ == '__main__':
    main()
