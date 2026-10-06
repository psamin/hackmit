import unittest

from interaction import InteractionTrack


class InteractionTests(unittest.TestCase):
    def sequence(self, track, start, count, contact, moving, valid=True, step=.2):
        found = []
        for index in range(count):
            found.extend(track.observe(start + index * step, contact, moving, valid))
        return found

    def test_shelf_and_background_motion_never_place(self):
        track = InteractionTrack(10)
        self.assertEqual(self.sequence(track, 10, 10, False, False), [])
        self.assertEqual(self.sequence(track, 12, 10, False, True), [])
        self.assertEqual(self.sequence(track, 14, 10, False, False), [])

    def test_pickup_then_release_rest_produces_two_distinct_events(self):
        track = InteractionTrack(10)
        self.sequence(track, 10, 5, False, False)
        events = self.sequence(track, 11, 6, True, True)
        self.assertEqual([e["event_type"] for e in events], ["picked_up"])
        events += self.sequence(track, 12.2, 5, False, False)
        self.assertEqual([e["event_type"] for e in events], ["picked_up", "placed"])
        self.assertEqual(events[-1]["evidence"]["contact_candidate"], True)
        self.assertTrue(events[-1]["evidence"]["released"])
        self.assertLess(events[-1]["window_start_at"], events[-1]["observed_at"])
        self.assertEqual(self.sequence(track, 13.2, 20, False, False), [])

    def test_first_seen_in_hand_can_place_but_does_not_invent_a_pickup(self):
        track = InteractionTrack(10)
        self.assertEqual(self.sequence(track, 10, 6, True, True), [])
        events = self.sequence(track, 11.2, 6, False, False)
        self.assertEqual([e["event_type"] for e in events], ["placed"])

    def test_contact_without_motion_is_not_placement(self):
        track = InteractionTrack(10)
        self.sequence(track, 10, 5, True, False)
        self.assertEqual(self.sequence(track, 11, 10, False, False), [])

    def test_invalid_ego_motion_never_counts_as_rest(self):
        track = InteractionTrack(10)
        self.sequence(track, 10, 6, True, True)
        self.assertEqual(self.sequence(track, 11.2, 6, False, False, valid=False), [])

    def test_track_gap_changes_identity_segment_and_discards_incomplete_action(self):
        track = InteractionTrack(10)
        self.sequence(track, 10, 6, True, True)
        segment = track.segment_id
        self.assertEqual(self.sequence(track, 20, 6, False, False), [])
        self.assertNotEqual(track.segment_id, segment)
        self.assertEqual(track.started_at, 20)

    def test_stationary_held_object_does_not_trigger_placement(self):
        track = InteractionTrack(10)
        self.sequence(track, 10, 6, True, True)
        self.assertEqual(self.sequence(track, 11.2, 10, True, False), [])
        self.assertEqual([e["event_type"] for e in self.sequence(track, 13.2, 6, False, False)], ["placed"])


if __name__ == "__main__":
    unittest.main()
