import unittest
from environment import CourtRoomEnv

class TestAsymmetricInformation(unittest.TestCase):
    def setUp(self):
        self.env = CourtRoomEnv(seed=42)
        self.env.reset()
        
    def test_prosecution_cannot_see_client_privilege(self):
        """
        Halluminate Bonus Test: Ensures the prosecution's observation
        does not contain the private client privileged information.
        """
        # Get prosecution's observation
        obs = self.env._build_observation(agent_id="prosecutor")
        
        # Verify the client privilege field is empty/masked
        self.assertEqual(obs.client_privilege, "", "Prosecution observation leaked client privilege!")
        
        # Verify prosecution CAN see the police FIR
        self.assertNotEqual(obs.police_fir, "", "Prosecution missing police FIR access!")

    def test_defense_cannot_see_police_fir_initially(self):
        """
        Ensures the defense does not see the raw police FIR initially,
        but can see their own client privilege.
        """
        # Get defense's observation
        obs = self.env._build_observation(agent_id="defense")
        
        # Verify the police FIR field is empty/masked
        self.assertEqual(obs.police_fir, "", "Defense observation leaked police FIR!")
        
        # Verify defense CAN see client privilege
        self.assertNotEqual(obs.client_privilege, "", "Defense missing client privilege access!")

    def test_judge_sees_only_on_record(self):
        """
        Ensures the judge sees neither side's private facts, only what is on record.
        """
        obs = self.env._build_observation(agent_id="judge")
        self.assertEqual(obs.client_privilege, "", "Judge observation leaked client privilege!")
        self.assertEqual(obs.police_fir, "", "Judge observation leaked police FIR!")

if __name__ == '__main__':
    unittest.main()
