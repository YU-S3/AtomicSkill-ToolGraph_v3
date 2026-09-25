"""Read-only public API probe; never generate gold or query hidden state."""
import json
from scienceworld import ScienceWorldEnv

def main():
    env = ScienceWorldEnv(envStepLimit=100)
    try:
        env.load('boil', 0, 'easy', generateGoldPath=False)
        observation, info = env.reset()
        print(json.dumps({'tasks': env.tasks, 'templates': env.get_possible_actions_with_IDs(),
            'observation': observation, 'info': {k:v for k,v in info.items() if k != 'valid'},
            'catalog': env.get_valid_action_object_combinations_with_templates()[:20],
            'simplifications': env.get_simplifications_used()}, indent=2), flush=True)
    finally:
        env.close()

if __name__ == '__main__':
    main()
