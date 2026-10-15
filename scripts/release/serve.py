"""Serve verified release models, PostgreSQL references and frontend on localhost.

Do not run while the main GPU benchmark/training owns the GPU. This launcher
does not download models, provide demo outputs or fall back to pilot boundaries.
"""
import argparse
from pathlib import Path
import sys
import ssl

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))


def server_options(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,help='Default HTTP port 8767; HTTPS uses separate port 8768')
    parser.add_argument('--https',action='store_true',help='Opt in to direct Uvicorn TLS on loopback port 8768')
    parser.add_argument('--ssl-certfile',type=Path)
    parser.add_argument('--ssl-keyfile',type=Path)
    parser.add_argument('--validate-only',action='store_true',help='Check release manifests/hashes without loading models or using GPU')
    args=parser.parse_args(argv)
    port=args.port if args.port is not None else (8768 if args.https else 8767)
    if not 1<=port<=65535:parser.error('port must be in 1..65535')
    if args.https and port!=8768:parser.error('HTTPS uses separate loopback port 8768; HTTP coordinator remains on 8767')
    if not args.https and (args.ssl_certfile or args.ssl_keyfile):parser.error('TLS files require --https')
    options=dict(host='127.0.0.1',port=port,workers=1,access_log=False,log_level='warning',timeout_keep_alive=5)
    if args.https:
        if bool(args.ssl_certfile)!=bool(args.ssl_keyfile):parser.error('Provide both --ssl-certfile and --ssl-keyfile')
        if args.ssl_certfile is None:
            from scripts.release.setup_local_tls import certificate_paths
            args.ssl_certfile,args.ssl_keyfile=certificate_paths()
        if not args.ssl_certfile.is_file() or not args.ssl_keyfile.is_file():
            raise RuntimeError('TLS files missing; run scripts/release/setup_local_tls.py first. No HTTP fallback.')
        ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER).load_cert_chain(str(args.ssl_certfile),str(args.ssl_keyfile))
        options.update(ssl_certfile=str(args.ssl_certfile),ssl_keyfile=str(args.ssl_keyfile))
    return args,options


def main():
    args,options=server_options()
    from facework.release_runtime import make_app,validate_deployment
    if args.validate_only:
        registry,manifest,depth=validate_deployment(ROOT)
        print(f"Validated {len(registry['features'])} categorical heads, {len(manifest['weights'])} weights, depth cohort {depth['cohort_n']}")
        return
    if not (ROOT/'frontend/dist/index.html').is_file():
        raise RuntimeError('Build frontend/dist before starting the release service')
    import uvicorn
    uvicorn.run(make_app(root=ROOT),**options)


if __name__=='__main__': main()
