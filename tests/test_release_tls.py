"""CPU-only TLS tests. Never import the release runtime or start its service."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from cryptography import x509
from scripts.release import serve,setup_local_tls


class ReleaseTLSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base=Path(os.environ['LOCALAPPDATA'])/'FaceTeller'
        base.mkdir(parents=True,exist_ok=True)
        cls.private=tempfile.TemporaryDirectory(prefix='tls-cpu-test-',dir=base)
        cls.environment=patch.dict(os.environ,{'LOCALAPPDATA':cls.private.name})
        cls.environment.start()
        cls.info=setup_local_tls.ensure_certificate()
        cls.cert=Path(cls.info['certfile']);cls.key=Path(cls.info['keyfile'])

    @classmethod
    def tearDownClass(cls):
        cls.environment.stop();cls.private.cleanup()

    def test_default_http_and_explicit_https_options(self):
        args,options=serve.server_options([])
        self.assertFalse(args.https)
        self.assertEqual(options['host'],'127.0.0.1')
        self.assertEqual(options['port'],8767)
        self.assertNotIn('ssl_certfile',options)
        args,options=serve.server_options(['--https'])
        self.assertTrue(args.https)
        self.assertEqual(options['host'],'127.0.0.1')
        self.assertEqual(options['port'],8768)
        self.assertEqual(options['ssl_certfile'],str(self.cert))
        self.assertEqual(options['ssl_keyfile'],str(self.key))
        self.assertEqual(serve.server_options(['--port','9000'])[1]['port'],9000)

    def test_tls_arguments_cannot_silently_fall_back_or_use_http_coordinator_port(self):
        for args in (['--https','--port','8767'],['--ssl-certfile',str(self.cert)],
                     ['--https','--ssl-keyfile',str(self.key)],['--port','0']):
            with self.subTest(args=args),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                serve.server_options(args)
        with self.assertRaisesRegex(RuntimeError,'missing'):
            serve.server_options(['--https','--ssl-certfile',str(self.cert)+'-missing','--ssl-keyfile',str(self.key)])

    def test_generation_is_private_san_scoped_and_repeatable(self):
        self.assertTrue(self.cert.is_relative_to(Path(self.private.name)))
        before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in (self.cert,self.key)]
        reused=setup_local_tls.ensure_certificate()
        self.assertFalse(reused['created'])
        self.assertFalse(reused['global_trust_modified'])
        self.assertEqual(before,[hashlib.sha256(p.read_bytes()).hexdigest() for p in (self.cert,self.key)])
        cert=x509.load_pem_x509_certificate(self.cert.read_bytes())
        self.assertEqual(cert.subject,cert.issuer)
        san=cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        self.assertEqual(san.get_values_for_type(x509.DNSName),['localhost'])
        self.assertEqual([str(ip) for ip in san.get_values_for_type(x509.IPAddress)],['127.0.0.1'])
        self.assertFalse(cert.extensions.get_extension_for_class(x509.BasicConstraints).value.ca)
        self.assertLessEqual((cert.not_valid_after_utc-cert.not_valid_before_utc).days,30)

    def test_missing_localappdata_never_generates_in_repository(self):
        with patch.dict(os.environ,{'LOCALAPPDATA':''}),self.assertRaisesRegex(RuntimeError,'LOCALAPPDATA'):
            setup_local_tls.ensure_certificate()

    def test_partial_pair_requires_explicit_renewal(self):
        with tempfile.TemporaryDirectory(dir=self.private.name) as folder,patch.dict(os.environ,{'LOCALAPPDATA':folder}):
            cert,key=setup_local_tls.certificate_paths();cert.parent.mkdir(parents=True)
            cert.write_bytes(self.cert.read_bytes())
            with self.assertRaisesRegex(RuntimeError,'Incomplete'):setup_local_tls.ensure_certificate()
            self.assertFalse(key.exists())
            renewed=setup_local_tls.ensure_certificate(renew=True)
            self.assertTrue(renewed['created'])
            self.assertTrue(key.is_file())

    def test_mismatched_key_is_rejected_before_runtime_import(self):
        with tempfile.TemporaryDirectory(dir=self.private.name) as folder,patch.dict(os.environ,{'LOCALAPPDATA':folder}):
            other=setup_local_tls.ensure_certificate()
            with self.assertRaises(ssl.SSLError):
                serve.server_options(['--https','--ssl-certfile',str(self.cert),'--ssl-keyfile',other['keyfile']])

    def test_cpu_entry_import_does_not_import_models_or_torch(self):
        code="import sys; from scripts.release import serve,setup_local_tls; assert 'torch' not in sys.modules; assert 'facework.release_runtime' not in sys.modules"
        result=subprocess.run([sys.executable,'-B','-c',code],cwd=serve.ROOT,capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)

    @unittest.skipUnless(os.name=='nt','Windows launcher parser')
    def test_powershell_launcher_parses_and_rejects_https_8767_before_preflight(self):
        script=serve.ROOT/'start-release.ps1'
        parse="$t=$null;$e=$null;[System.Management.Automation.Language.Parser]::ParseFile($args[0],[ref]$t,[ref]$e)>$null;if($e.Count){exit 1}"
        # A literal PowerShell path is quoted separately, never interpolate secrets.
        escaped=str(script).replace("'","''")
        result=subprocess.run(['powershell.exe','-NoProfile','-Command',parse.replace('$args[0]',f"'{escaped}'")],capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)
        result=subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(script),'-Https','-Port','8767'],capture_output=True,text=True,timeout=15)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('separate loopback port 8768',result.stderr)

    def test_actual_uvicorn_tls_handshake_and_request_without_service_or_gpu(self):
        import uvicorn
        async def tiny(scope,receive,send):
            self.assertEqual(scope['scheme'],'https')
            await send({'type':'http.response.start','status':200,'headers':[(b'content-type',b'application/json')]})
            await send({'type':'http.response.body','body':b'{"cpu_tls":true}'})
        # An OS-assigned ephemeral listener, never production ports 8767/8768.
        listener=socket.socket();listener.bind(('127.0.0.1',0))
        port=listener.getsockname()[1];self.assertNotIn(port,(8767,8768))
        _,options=serve.server_options(['--https'])
        options['port']=0
        config=uvicorn.Config(tiny,**options,lifespan='off',http='h11',ws='none',log_config=None)
        server=uvicorn.Server(config)
        worker=threading.Thread(target=server.run,kwargs={'sockets':[listener]},daemon=True)
        worker.start()
        try:
            deadline=time.monotonic()+10
            while not server.started and worker.is_alive() and time.monotonic()<deadline:time.sleep(.01)
            self.assertTrue(server.started,'CPU TLS listener did not start')
            def fetch(context,hostname):
                with socket.create_connection(('127.0.0.1',port),timeout=3) as raw:
                    with context.wrap_socket(raw,server_hostname=hostname) as connection:
                        connection.sendall(f'GET / HTTP/1.1\r\nHost: {hostname}\r\nConnection: close\r\n\r\n'.encode())
                        response=b''
                        while data:=connection.recv(4096):response+=data
                        return response
            # Only this client trusts the supplied leaf; verification stays enabled.
            trusted=ssl.create_default_context(cafile=str(self.cert))
            for hostname in ('localhost','127.0.0.1'):
                with self.subTest(hostname=hostname):
                    response=fetch(trusted,hostname)
                    self.assertIn(b'200 OK',response)
                    self.assertIn(b'{"cpu_tls":true}',response)
            with self.assertRaises(ssl.SSLCertVerificationError):fetch(trusted,'not-localhost.invalid')
            with self.assertRaises(ssl.SSLCertVerificationError):fetch(ssl.create_default_context(),'localhost')
        finally:
            server.should_exit=True;worker.join(timeout=10);listener.close()
        self.assertFalse(worker.is_alive(),'CPU TLS listener was not cleaned up')


if __name__=='__main__':unittest.main()
