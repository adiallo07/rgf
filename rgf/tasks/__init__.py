def get_task(args):
    if args.task == 'game24':
        from tasks.game24 import Game24Task
        return Game24Task(args)
    elif args.task == 'penguin':
        from tasks.penguin import PenguinInATableTask
        return PenguinInATableTask(args)
    elif args.task == 'checkmate':
        from tasks.checkmateone import CheckmateInOneTask
        return CheckmateInOneTask(args)
    elif args.task == 'sonnet':
        from tasks.sonnet import SonnetWritingTask
        return SonnetWritingTask(args)
    elif args.task == 'crosswords':
        from tasks.crosswords import MiniCrosswordsTask
        return MiniCrosswordsTask(args)
    elif args.task == 'puzzle':
        from tasks.puzzle import PythonPuzzleTask
        return PythonPuzzleTask(args)
    else:
        raise NotImplementedError


