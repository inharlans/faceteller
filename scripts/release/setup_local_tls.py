"""Create/reuse a private, self-signed localhost certificate; never install trust.

This is local development TLS, not a public certificate or remote deployment.
Only %LOCALAPPDATA%/FaceTeller/tls is used; no certificate/key enters the project.
"""
import argparse
import csv
import datetime
import ipaddress
import json
import os
from pathlib import Path
import ssl
import subprocess


def local_tls_directory():
    value=os.environ.get('LOCALAPPDATA')
    if not value or not Path(value).is_absolute():
        raise RuntimeError('An absolute LOCALAPPDATA directory is required for private local TLS files')
    return Path(value)/'FaceTeller'/'tls'


def certificate_paths():
    directory=local_tls_directory()
    return directory/'localhost-cert.pem',directory/'localhost-key.pem'


def restrict_directory(directory):
    """Remove inherited access on this TLS directory only, before writing a key."""
    directory.mkdir(parents=True,exist_ok=True)
    if directory.is_symlink():raise RuntimeError('TLS directory must not be a symlink')
    if os.name=='nt':
        identity=subprocess.check_output(['whoami','/user','/fo','csv','/nh'],text=True)
        sid=next(csv.reader(identity.strip().splitlines()))[-1]
        if not sid.startswith('S-1-'):raise RuntimeError('Cannot determine current Windows user SID')
        subprocess.run(['icacls',str(directory),'/inheritance:r','/grant:r',
                        f'*{sid}:(OI)(CI)F','*S-1-5-18:(OI)(CI)F'],check=True,
                       stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    else:
        directory.chmod(0o700)


def ensure_certificate(*,renew=False):
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes,serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import ExtendedKeyUsageOID,NameOID
    except ImportError as exc:
        raise RuntimeError('Local certificate generation requires cryptography; install requirements-release.txt in the Windows .venv') from exc
    cert_path,key_path=certificate_paths()
    restrict_directory(cert_path.parent)
    now=datetime.datetime.now(datetime.timezone.utc)
    if (cert_path.exists() or key_path.exists()) and not renew:
        if not cert_path.is_file() or not key_path.is_file():
            raise RuntimeError('Incomplete local TLS pair; explicitly run setup_local_tls.py --renew')
        cert=x509.load_pem_x509_certificate(cert_path.read_bytes())
        san=cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        if not (cert.not_valid_before_utc<=now<cert.not_valid_after_utc):
            raise RuntimeError('Local certificate is not currently valid; explicitly run setup_local_tls.py --renew')
        if 'localhost' not in san.get_values_for_type(x509.DNSName) or ipaddress.ip_address('127.0.0.1') not in san.get_values_for_type(x509.IPAddress):
            raise RuntimeError('Local certificate must cover localhost and 127.0.0.1')
        ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER).load_cert_chain(str(cert_path),str(key_path))
        created=False
    else:
        key=rsa.generate_private_key(public_exponent=65537,key_size=2048)
        name=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,'FaceTeller localhost development')])
        cert=(x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
              .serial_number(x509.random_serial_number()).not_valid_before(now-datetime.timedelta(minutes=5))
              .not_valid_after(now+datetime.timedelta(days=30))
              .add_extension(x509.SubjectAlternativeName([x509.DNSName('localhost'),x509.IPAddress(ipaddress.ip_address('127.0.0.1'))]),critical=False)
              .add_extension(x509.BasicConstraints(ca=False,path_length=None),critical=True)
              .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),critical=False)
              .add_extension(x509.KeyUsage(digital_signature=True,content_commitment=False,key_encipherment=True,
                                          data_encipherment=False,key_agreement=False,key_cert_sign=False,
                                          crl_sign=False,encipher_only=False,decipher_only=False),critical=True)
              .sign(key,hashes.SHA256()))
        payloads=((key_path,key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption())),
                  (cert_path,cert.public_bytes(serialization.Encoding.PEM)))
        for path,payload in payloads:
            temporary=path.with_suffix('.tmp')
            try:
                with temporary.open('wb') as stream:
                    if os.name!='nt':temporary.chmod(0o600)
                    stream.write(payload)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER).load_cert_chain(str(cert_path),str(key_path))
        created=True
    return {'certfile':str(cert_path),'keyfile':str(key_path),'created':created,
            'expires_utc':cert.not_valid_after_utc.isoformat(),'self_signed_development':True,
            'global_trust_modified':False,'url':'https://127.0.0.1:8768/'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--renew',action='store_true',help='Explicitly replace the local development certificate/key')
    args=parser.parse_args()
    print(json.dumps(ensure_certificate(renew=args.renew),ensure_ascii=False))


if __name__=='__main__':main()
