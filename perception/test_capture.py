import math
import unittest

from capture import decode_frame, encode_frame, location_at


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.fix = {"lat": 40.0, "lon": -71.0, "accuracy_m": 12.0, "observed_at": 1000.0}
        self.metadata = {"session_id": "camera-test-1", "frame_id": 1,
                         "captured_at": 1010.0, "location": self.fix}

    def test_packet_preserves_frame_and_capture_location(self):
        jpeg = b"\xff\xd8test-jpeg\xff\xd9"
        image, meta = decode_frame(encode_frame(jpeg, self.metadata), received_at=1011.0)
        self.assertEqual(image, jpeg)
        self.assertEqual(meta["captured_at"], 1010.0)
        self.assertEqual(meta["location"]["observed_at"], 1000.0)
        self.assertEqual(meta["location"]["accuracy_m"], 12.0)
        self.assertEqual(meta["location"]["status"], "available")
        self.assertIn("query=40.000000,-71.000000", meta["location"]["maps_url"])

    def test_legacy_frames_cannot_inherit_location(self):
        image, meta = decode_frame(b"legacy-frame", received_at=1011.0)
        self.assertEqual(image, b"legacy-frame")
        self.assertEqual(meta["captured_at"], 1011.0)
        self.assertEqual(meta["time_source"], "received")
        self.assertEqual(meta["location"]["status"], "missing")

    def test_stale_future_and_poor_fixes_have_no_map(self):
        cases = [({**self.fix, "observed_at": 800.0}, "stale"),
                 ({**self.fix, "observed_at": 1011.0}, "future"),
                 ({**self.fix, "accuracy_m": 300.0}, "inaccurate"),
                 ({"lat": 40.0, "lon": -71.0}, "invalid")]
        for fix, reason in cases:
            with self.subTest(reason=reason):
                loc = location_at(fix, 1010.0)
                self.assertEqual(loc["status"], reason)
                self.assertNotIn("maps_url", loc)
                self.assertNotIn("lat", loc)

    def test_nonfinite_and_out_of_range_locations_rejected(self):
        for key, value in [("lat", math.nan), ("lon", math.inf), ("lat", 91),
                           ("lon", -181), ("accuracy_m", -1), ("lat", True)]:
            with self.subTest(key=key, value=value):
                self.assertEqual(location_at({**self.fix, key: value}, 1010.0)["status"], "invalid")

    def test_missing_permission_is_not_home(self):
        self.assertEqual(location_at(None, 1010.0), {"status": "missing"})

    def test_delayed_packet_rejected_instead_of_retimestamped(self):
        data = encode_frame(b"frame", self.metadata)
        with self.assertRaises(ValueError):
            decode_frame(data, received_at=1100.0)
        with self.assertRaises(ValueError):
            decode_frame(data, received_at=900.0)

    def test_malformed_metadata_and_oversized_packets_rejected(self):
        for data in [b"CMP2", b"CMP2\x00\x00\x00\x03{}", b"CMP2\x00\x00\x00\x01[]"]:
            with self.subTest(data=data), self.assertRaises(ValueError):
                decode_frame(data, received_at=1011.0)
        with self.assertRaises(ValueError):
            encode_frame(b"x" * (2 ** 20), self.metadata)

    def test_invalid_sequence_rejected(self):
        for value in [-1, True, "one"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                decode_frame(encode_frame(b"frame", {**self.metadata, "frame_id": value}), received_at=1011.0)


if __name__ == "__main__":
    unittest.main()
