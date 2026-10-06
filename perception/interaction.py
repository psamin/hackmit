from __future__ import annotations

import uuid


class InteractionTrack:
    def __init__(self, at, contact_motion_s=.3, rest_s=.6, max_gap_s=.8):
        self.contact_motion_s, self.rest_s, self.max_gap_s = contact_motion_s, rest_s, max_gap_s
        self.reset(at)

    def reset(self, at):
        self.segment_id = str(uuid.uuid4())
        self.started_at, self.last_seen_at = at, None
        self.contact_start = self.last_contact = self.release_at = self.rest_start = None
        self.contact_motion = 0.0
        self.armed = self.pickup_emitted = self.saw_rest = False

    def observe(self, at, contact_candidate, moving, valid_motion):
        if self.last_seen_at is not None and (at <= self.last_seen_at or at - self.last_seen_at > self.max_gap_s):
            self.reset(at)
        dt = 0.0 if self.last_seen_at is None else at - self.last_seen_at
        self.last_seen_at = at
        events = []
        if not valid_motion:
            self.rest_start = None
            return events
        if contact_candidate:
            self.contact_start = at if self.contact_start is None else self.contact_start
            self.last_contact, self.release_at, self.rest_start = at, None, None
            if moving:
                self.contact_motion += dt
                self.armed = self.armed or self.contact_motion + 1e-8 >= self.contact_motion_s
                if self.armed and self.saw_rest and not self.pickup_emitted:
                    events.append(self.proposal("picked_up", at))
                    self.pickup_emitted = True
        else:
            if self.last_contact is not None and self.release_at is None:
                self.release_at = at
            if moving:
                self.rest_start = None
            else:
                self.rest_start = at if self.rest_start is None else self.rest_start
                if at - self.rest_start + 1e-8 >= self.rest_s:
                    if self.armed and self.release_at is not None and at - self.last_contact <= 6.0:
                        events.append(self.proposal("placed", at))
                        self.armed = self.pickup_emitted = False
                        self.contact_start = self.last_contact = self.release_at = None
                        self.contact_motion = 0.0
                    self.saw_rest = True
            if self.last_contact is not None and at - self.last_contact > 6.0:
                self.armed = self.pickup_emitted = False
                self.contact_start = self.last_contact = self.release_at = None
                self.contact_motion = 0.0
        return events

    def proposal(self, event_type, at):
        return {"event_type": event_type, "observed_at": at,
                "window_start_at": max(self.started_at, (self.contact_start if self.contact_start is not None else at) - 1.0),
                "contact_at": self.last_contact,
                "evidence": {"contact_candidate": True, "motion_valid": True,
                             "released": event_type == "placed", "track_continuous": True,
                             "track_started_at": self.started_at, "contact_started_at": self.contact_start,
                             "contact_ended_at": self.last_contact, "release_at": self.release_at,
                             "contact_duration_s": self.contact_motion, "contact_source": "person_box_overlap"}}
