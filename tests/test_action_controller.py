import unittest

from src.action_controller import ActionController


class ActionControllerGestureStabilityTests(unittest.TestCase):
    def test_discrete_action_requires_repeated_gesture_frames(self):
        controller = object.__new__(ActionController)
        controller.last_gesture = None
        controller.gesture_streak = 0

        self.assertFalse(controller._should_trigger_discrete_action("Five Fingers"))

        controller.last_gesture = "Five Fingers"
        controller.gesture_streak = 1
        self.assertFalse(controller._should_trigger_discrete_action("Five Fingers"))

        controller.gesture_streak = 2
        self.assertTrue(controller._should_trigger_discrete_action("Five Fingers"))

    def test_discrete_action_resets_when_gesture_changes(self):
        controller = object.__new__(ActionController)
        controller.last_gesture = "Five Fingers"
        controller.gesture_streak = 2

        self.assertFalse(controller._should_trigger_discrete_action("Heart Gesture"))
        self.assertEqual(controller.last_gesture, "Heart Gesture")
        self.assertEqual(controller.gesture_streak, 1)


if __name__ == "__main__":
    unittest.main()
