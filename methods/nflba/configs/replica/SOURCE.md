Copied unchanged from MonoGS (muskie82/MonoGS @ 6c9254c319d8bff5caeef65259e6bb0941a9b9f6,
configs/rgbd/replica/), the base NFL-BA builds on. NFL-BA ships no Replica config.
methods/nflba/adapter.py merges <scene>.yaml over base_config.yaml (inherit_from, as
MonoGS's utils/config_utils.load_config does) and applies the overrides it records in
overrides.json.
