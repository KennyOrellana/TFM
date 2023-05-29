from core.base_orchestrator import BaseOrchestrator


def run():
    orchestrator = BaseOrchestrator()
    orchestrator.execute('example/demo4.json')


if __name__ == '__main__':
    run()
