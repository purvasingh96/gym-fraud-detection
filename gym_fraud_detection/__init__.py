from gym.envs.registration import register

register(
    id='gym-fraud-detection-v0',
    entry_point='gym_fraud_detection.envs:FraudDetectionEnv',
)