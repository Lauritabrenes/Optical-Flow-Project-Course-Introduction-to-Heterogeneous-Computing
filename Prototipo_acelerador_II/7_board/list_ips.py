from pynq import Overlay
ol = Overlay('lk_v0.bit')
for k, v in ol.ip_dict.items():
    addr = v.get('phys_addr', v.get('base_address'))
    print(f"\n{k}  type={v.get('type')}  addr={hex(addr) if addr is not None else 'n/a'}")
    for r, d in (v.get('registers') or {}).items():
        print(f"    {r:12s} {hex(d.get('address_offset', 0))}")
