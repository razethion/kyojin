"""Restrict optional direct worker binding to an actual local USB4 netdev."""
import ipaddress
import json
import re
import subprocess
from pathlib import Path


def validate_binding(host, interface=None, *, inventory=None, device_path=None):
    address = ipaddress.ip_address(host.split('%', 1)[0])
    if interface is None:
        if not address.is_loopback:
            raise ValueError('non-loopback workers require an explicit USB4 interface')
        return
    if not re.fullmatch(r'thunderbolt[0-9]+', interface):
        raise ValueError('expected a thunderbolt network interface')
    if '%' in host and host.split('%', 1)[1] != interface:
        raise ValueError('IPv6 address scope differs from USB4 interface')
    if address.is_loopback or address.is_unspecified or address.is_multicast or not (address.is_private or address.is_link_local):
        raise ValueError('direct binding requires a private point-to-point address')
    path = device_path or str((Path('/sys/class/net')/interface/'device').resolve())
    if not re.search(r'/domain[0-9]+/[0-9]+-0/[0-9]+-[0-9a-f]+/', path):
        raise ValueError('interface is not attached to a USB4 peer')
    if inventory is None:
        inventory = json.loads(subprocess.check_output(['ip','-j','address','show','dev',interface],text=True,timeout=5))
    owned = {item['local'].split('%',1)[0] for dev in inventory if dev.get('ifname')==interface for item in dev.get('addr_info',[])}
    if str(address) not in owned:
        raise ValueError('bind address is not configured on the selected USB4 interface')
