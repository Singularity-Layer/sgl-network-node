import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location("eg2_worker", Path(__file__).resolve().parents[1] / "scripts/embeddinggemma_worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)

class WorkerPolicy(unittest.TestCase):
    def test_duration_refuses_understatement_and_limits(self):
        worker.check_duration(1.0, 1.0, 30)
        for actual, declared in ((1.1,1.0), (31,31), (float('nan'),1), (0,0), (1,True)):
            with self.assertRaises(ValueError):
                worker.check_duration(actual,declared,30)

    def test_pinned_private_video_signature_and_metadata_check(self):
        calls = []
        def decode(path, sampler):
            calls.append(path)
            return sampler(SimpleNamespace(duration=2.0))
        processor = SimpleNamespace(video_processor=SimpleNamespace(_decode_video=decode, sample_frames=lambda metadata, **kw: [0,1]))
        self.assertEqual(worker.verify_video(processor,Path('/private/example.mp4'),2.0),[0,1])
        self.assertEqual(calls,['/private/example.mp4'])
        with self.assertRaises(ValueError):
            worker.verify_video(processor,Path('/private/example.mp4'),1.0)
        processor.video_processor.sample_frames = lambda metadata, **kw: list(range(33))
        with self.assertRaises(ValueError):
            worker.verify_video(processor,Path('/private/example.mp4'),2.0)

    def test_processed_budget_is_per_item_and_modalities_must_be_present(self):
        text = {"content":[{"type":"text"}]}
        rows = [worker.processed_usage(text,8192,0,0,0) for _ in range(2)]
        self.assertEqual(sum(sum(row.values()) for row in rows),16384)
        with self.assertRaises(ValueError):
            worker.processed_usage(text,8193,0,0,0)
        image = {"content":[{"type":"image"}]}
        with self.assertRaises(RuntimeError):
            worker.processed_usage(image,12,0,0,0)
        with self.assertRaises(RuntimeError):
            worker.processed_usage(image,300,281,0,0)
        self.assertEqual(worker.processed_usage(image,268,256,0,0),{"text":12,"image":256,"audio":0,"video":0})

    def test_processor_accounting_failures_are_fatal_but_context_limit_is_input(self):
        text={"content":[{"type":"text"}]}
        for active,image,audio,video in ((12,0,1,0),(12,0,0,1),(12,-1,0,0)):
            with self.assertRaises(RuntimeError):
                worker.processed_usage(text,active,image,audio,video)
        with self.assertRaises(ValueError):
            worker.processed_usage(text,8193,0,0,0)
        video={"content":[{"type":"video","duration_seconds":1}]}
        with self.assertRaises(RuntimeError):
            worker.processed_usage(video,153,0,0,141)
        audio={"content":[{"type":"audio"}]}
        self.assertEqual(worker.processed_usage(audio,37,0,25,0),{"text":12,"image":0,"audio":25,"video":0})

    def test_text_preflight_uses_utf8_prefix_and_template_bytes(self):
        item = lambda text: [{"content":[{"type":"text","text":text}]}]
        for input_type, prefix in worker.PREFIXES.items():
            allowed = 8192 - 12 - len(prefix.encode("utf-8"))
            worker.validate_text_budget(item("x"*allowed),input_type)
            with self.assertRaises(ValueError):
                worker.validate_text_budget(item("x"*(allowed+1)),input_type)
        worker.validate_text_budget(item("é"*4090),"unspecified")
        with self.assertRaises(ValueError):
            worker.validate_text_budget(item("é"*4091),"unspecified")
        for text in ("", " \t", "x"*1_000_000):
            with self.assertRaises(ValueError):
                worker.validate_text_budget(item(text),"unspecified")

    def test_large_text_fails_before_processor_or_media_work(self):
        runtime = object.__new__(worker.Runtime)
        runtime._conversations = lambda *args: self.fail("media preprocessing called")
        runtime.processor = SimpleNamespace(apply_chat_template=lambda *a,**kw:self.fail("tokenizer called"))
        request = {"type":"embed","protocol":worker.PROTOCOL,"request_id":1,"input_type":"query","input":[{"content":[{"type":"text","text":"x"*1_000_000}]}]}
        with self.assertRaises(ValueError):
            runtime.embed(request)

    def test_native_shape_finiteness_and_norm_failures_are_runtime_errors(self):
        class Predicate:
            def __init__(self,value): self.value=value
            def all(self): return self.value
            def any(self): return self.value
        class Norm:
            def __init__(self,value): self.value=value; self.finite=__import__('math').isfinite(value)
            def __le__(self,value): return Predicate(self.value<=value)
        class Vectors:
            def __init__(self,shape=(1,768),finite=True,native_norm=1.0,truncated_norm=1.0):
                self.shape=shape; self.finite=finite; self.native_norm=native_norm; self.truncated_norm=truncated_norm
            def __getitem__(self,key):
                result=Vectors((self.shape[0],key[1].stop),self.finite,self.native_norm,self.truncated_norm)
                return result
            def __itruediv__(self,other): return self
        np = SimpleNamespace(isfinite=lambda value:Predicate(value.finite),allclose=lambda norm,target,**kw:abs(norm.value-target)<=kw['atol'],linalg=SimpleNamespace(norm=lambda vector,axis,keepdims=False:Norm(vector.truncated_norm if keepdims else vector.native_norm)))
        for vectors in (Vectors(shape=(1,767)),Vectors(finite=False),Vectors(native_norm=0.5),Vectors(truncated_norm=0),Vectors(truncated_norm=float('inf'))):
            with self.assertRaises(RuntimeError):
                worker.normalize_native_output(vectors,1,128,np)
        self.assertEqual(worker.normalize_native_output(Vectors(),1,128,np).shape,(1,128))

    def test_expected_input_errors_do_not_terminate_protocol(self):
        class InvalidInput:
            def embed(self,request): raise worker.InputValidationError("private invalid input")
        for request_id in range(5):
            request={"type":"embed","protocol":worker.PROTOCOL,"request_id":request_id}
            self.assertEqual(worker.dispatch_request(InvalidInput(),request),{"type":"request_error","request_id":request_id,"code":"invalid_input"})
        class BrokenRuntime:
            def embed(self,request): raise RuntimeError("native failure")
        with self.assertRaises(RuntimeError):
            worker.dispatch_request(BrokenRuntime(),request)
        with self.assertRaises(RuntimeError):
            worker.dispatch_request(InvalidInput(),{"type":"embed","protocol":"corrupt","request_id":1})

    def test_mp4_brand_whitelist_refuses_mov_heif_and_3gp(self):
        def container(major,compatible):
            size=16+4*len(compatible)
            return size.to_bytes(4,'big')+b'ftyp'+major+b'\x00'*4+b''.join(compatible)
        worker.validate_mp4_container(container(b'isom',[b'iso2',b'avc1',b'mp41']))
        worker.validate_mp4_container((Path(__file__).resolve().parents[1]/'assets/embeddinggemma2/smoke/video.mp4').read_bytes())
        for raw in (container(b'qt  ',[b'isom']),container(b'heic',[b'isom']),container(b'3gp6',[b'isom']),container(b'isom',[b'qt  ']),b'\x00'*4+b'ftyp'+b'isom'+b'\x00'*4):
            with self.assertRaises(ValueError):
                worker.validate_mp4_container(raw)

    def test_library_value_error_is_fatal_not_input_frame(self):
        class BrokenLibrary:
            def embed(self,request): raise ValueError("unclassified library failure")
        request={"type":"embed","protocol":worker.PROTOCOL,"request_id":1}
        with self.assertRaises(RuntimeError):
            worker.dispatch_request(BrokenLibrary(),request)

    def test_preflight_includes_media_budget_before_processing(self):
        for part in ({"type":"image"},{"type":"audio","duration_seconds":1},{"type":"video","duration_seconds":1}):
            items=[{"content":[{"type":"text","text":"x"*8180},part]}]
            with self.assertRaises(worker.InputValidationError):
                worker.validate_text_budget(items,"unspecified")

    def test_bounded_audio_metadata_stream_and_bomb(self):
        import array
        class DecoderError(Exception): pass
        class FakeAudio:
            MiniaudioError=DecoderError
            FileFormat=SimpleNamespace(WAV='wav',FLAC='flac',MP3='mp3')
            SampleFormat=SimpleNamespace(FLOAT32='f32')
            def __init__(self,format='wav',declared=1,bomb=False):
                self.info=SimpleNamespace(file_format=format,nchannels=1,sample_rate=16000,num_frames=int(declared*16000),duration=declared)
                self.stream_calls=0; self.closed=False; self.bomb=bomb; self.chunks=0
            def get_file_info(self,path): return self.info
            def stream_file(self,path,**kwargs):
                self.stream_calls+=1
                self.kwargs=kwargs
                def chunks():
                    try:
                        remaining=31*16000 if self.bomb else self.info.num_frames
                        while remaining:
                            amount=min(4096,remaining); remaining-=amount; self.chunks+=1
                            yield array.array('f',[0.0])*amount
                    finally: self.closed=True
                return chunks()
        for mime,format in [('audio/wav','wav'),('audio/flac','flac'),('audio/mpeg','mp3')]:
            audio=FakeAudio(format)
            self.assertEqual(len(worker.decode_audio_bounded('/owned/input',mime,1,audio)),16000)
            self.assertEqual(audio.kwargs,{'output_format':'f32','nchannels':1,'sample_rate':16000,'frames_to_read':4096})
            self.assertTrue(audio.closed)
        audio=FakeAudio(declared=31)
        with self.assertRaises(worker.InputValidationError): worker.decode_audio_bounded('/owned/input','audio/wav',30,audio)
        self.assertEqual(audio.stream_calls,0)
        audio=FakeAudio(declared=30,bomb=True)
        with self.assertRaises(worker.InputValidationError): worker.decode_audio_bounded('/owned/input','audio/wav',30,audio)
        self.assertLessEqual(audio.chunks,118)
        self.assertTrue(audio.closed)
        audio=FakeAudio(declared=1,bomb=True)
        with self.assertRaises(worker.InputValidationError): worker.decode_audio_bounded('/owned/input','audio/wav',1,audio)
        self.assertLessEqual(audio.chunks,4)
        self.assertTrue(audio.closed)
        audio=FakeAudio('flac')
        with self.assertRaises(worker.InputValidationError): worker.decode_audio_bounded('/owned/input','audio/mpeg',1,audio)
        self.assertEqual(audio.stream_calls,0)

    def test_parameters_reject_float16_and_empty_models(self):
        mx = SimpleNamespace(bfloat16='bf16',float32='fp32')
        flatten = lambda parameters: parameters
        worker.verify_parameter_dtypes([('weights',SimpleNamespace(dtype='bf16')),('norm',SimpleNamespace(dtype='fp32'))], mx, flatten)
        for values in ([], [('weights',SimpleNamespace(dtype='fp16'))]):
            with self.assertRaises(RuntimeError):
                worker.verify_parameter_dtypes(values,mx,flatten)

if __name__ == '__main__':
    unittest.main()
