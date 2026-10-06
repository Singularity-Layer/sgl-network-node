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
        with self.assertRaises(ValueError):
            worker.processed_usage(image,12,0,0,0)
        with self.assertRaises(ValueError):
            worker.processed_usage(image,300,281,0,0)
        self.assertEqual(worker.processed_usage(image,268,256,0,0),{"text":12,"image":256,"audio":0,"video":0})

    def test_parameters_reject_float16_and_empty_models(self):
        mx = SimpleNamespace(bfloat16='bf16',float32='fp32')
        flatten = lambda parameters: parameters
        worker.verify_parameter_dtypes([('weights',SimpleNamespace(dtype='bf16')),('norm',SimpleNamespace(dtype='fp32'))], mx, flatten)
        for values in ([], [('weights',SimpleNamespace(dtype='fp16'))]):
            with self.assertRaises(ValueError):
                worker.verify_parameter_dtypes(values,mx,flatten)

if __name__ == '__main__':
    unittest.main()
