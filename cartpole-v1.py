import gym
import random

env = gym.make('CartPole-v1')                                               # 'CartPole-v1' 환경 생성          
goal_steps = 500                                                                         # 최대 시행 횟수 (시간 단계) 설정 : 하나의 에피소드에서 cart가 움직일 수 있는 최대 시간

while True:                                                                                     # 무한 루프
    obs = env.reset()                                                                      # 환경 초기화
    total_reward = 0		                               # 에피소드 총 보상 초기화

    for i in range(goal_steps):                                                      # 최대 goal_step만큼 반복
        action = random.randrange(0, 2)                                     # 무작위 액션으로 0, 1을 선택
        obs, reward, done, info = env.step(action)                     # 무작위 액션에 따라 얻어지는 다음 상태, 보상, 종료 여부를 획득
        total_reward += reward                                                       # 보상 누적 

        if done:                                                                                     # 종료 여부에 대한 정보를 획득하게 될 경우,
            print(f"Episode reward: {total_reward}")                    # 에피소드 보상 출력
            break                                                                                     

        env.render()                                                                             # 환경 시각화

