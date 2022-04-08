# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import threading
import math
import time

@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.autoplay = True
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.max_depth = 0
        self.default_max_val = -math.inf
        self.default_min_val = math.inf
        self.threadLock = threading.Lock()
        self.max_val = self.default_max_val
        self.best_move = None
        self.alpha = self.default_max_val
        self.beta = self.default_min_val

    def solve(self, m, b, chess_board, my_pos, adv_pos, max_step):
        # print ("threading", m, b, my_pos, adv_pos, max_step)
        state = self.get_next_state(chess_board, my_pos, adv_pos, 0, m, b)
        if state is None:
            return

        alpha = self.alpha
        beta = self.beta

        value, _ = self.get_min(state, 0, max_step, alpha, beta)
        # print (m, b, value, self.max_val)

        self.threadLock.acquire()
        if value is not None and value > self.max_val:
            self.max_val = value
            self.best_move = m, b

        self.threadLock.release()


    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        self.max_val = self.default_max_val
        self.best_move = None

        all_move = self.get_all_moves(chess_board, my_pos, adv_pos, max_step)
        #print ("moves: ", len(all_move))

        for m, b in all_move:
            t = threading.Thread(target=self.solve,args=(m, b, chess_board, my_pos, adv_pos, max_step))
            t.start()
            t.join()
            # self.solve(m, b, chess_board, my_pos, adv_pos, max_step)

        if self.best_move is None:
            for m, b in all_move:
                state = self.get_next_state(chess_board, my_pos, adv_pos, 0, m, b)
                if state is not None:
                    return m, b

            return all_move[0]

        # bestm, bestb = self.best_move
        return self.best_move
        #print ("best: ", bestm, bestb, max_val)
        # best_state = self.get_next_state(chess_board, my_pos, adv_pos, 0, bestm, bestb)
        # # x, y = self.move(my_pos, bestm)
        # # print(bestm, bestb, x, y, chess_board[x, y, bestb], my_pos, adv_pos, best_state)
        # if best_state is None:
        #     return my_pos, bestb
        #
        # c, my, adv = best_state
        #
        # return my, bestb

    def available_pos(self, chess_board, start_pos, adv_pos, max_step):
        res = []
        steps_map = self.get_step_map(chess_board, start_pos, adv_pos)
        for i in range(len(steps_map)):
            for j in range(len(steps_map)):
                if steps_map[i][j] <= max_step:
                    res.append((i, j))

        return res

    def get_step_map(self, chess_board, start_pos, adv_pos):
        # BFS
        x, y = start_pos
        state_queue = [(x, y, 0)]
        visited = [(x, y)]
        steps = []
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        l = len(chess_board)

        for i in range(l):
            steps.append([])
            for _ in range(l):
                steps[i].append(math.inf)

        steps[x][y] = 0
        # print(chess_board)
        # print(steps)

        while len(state_queue) > 0:
            x, y, cur_step = state_queue.pop(0)

            for m in moves:
                dx, dy = m
                a = dx + x
                b = dy + y
                next_pos = a, b

                if next_pos == adv_pos:
                    continue

                if a >= l or a < 0:
                    continue
                if b >= l or b < 0:
                    continue
                if dx == -1 and chess_board[x, y, 0]:
                    continue
                if dy == 1 and chess_board[x, y, 1]:
                    continue
                if dx == 1 and chess_board[x, y, 2]:
                    continue
                if dy == -1 and chess_board[x, y, 3]:
                    continue

                if next_pos in visited:
                    continue

                visited.append(next_pos)
                state_queue.append((a, b, cur_step + 1))
                steps[a][b] = cur_step + 1

        return steps

    def get_next_state(self, chess_board, my_pos, adv_pos, turn, m, b):
        # Moves (Up, Right, Down, Left)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        # Opposite Directions
        opposites = {0: 2, 1: 3, 2: 0, 3: 1}

        if turn == 0:
            my_pos_new = m
            if my_pos_new == adv_pos:
                return None

            r, c = my_pos_new

            if r >= len(chess_board) or r < 0:
                # print(r, len(chess_board))
                return None
            if c >= len(chess_board) or c < 0:
                # print(c, len(chess_board))
                return None
            if chess_board[r, c, b]:
                # print(r, c, b)
                return None

            chess_board_new = chess_board.copy()
            chess_board_new[r, c, b]= True
            move = moves[b]
            chess_board_new[r + move[0], c + move[1], opposites[b]] = True

            return chess_board_new, my_pos_new, adv_pos

        else:
            adv_pos_new = m
            if my_pos == adv_pos_new:
                return None

            r, c = adv_pos_new
            if r >= len(chess_board) or r < 0:
                # print(r, len(chess_board))
                return None
            if c >= len(chess_board) or c < 0:
                # print(c, len(chess_board))
                return None
            if chess_board[r, c, b]:
                # print(r, c, b)
                return None

            chess_board_new = chess_board.copy()
            chess_board_new[r, c, b] = True
            move = moves[b]
            chess_board_new[r + move[0], c + move[1], opposites[b]] = True

            return chess_board_new, my_pos, adv_pos_new

    def get_all_moves(self, chess_board, start_pos, adv_pos, max_steps):
        move_direction = self.available_pos(chess_board, start_pos, adv_pos, max_steps)
        barrier_direction = (0, 1, 2, 3)
        res = []
        for m in move_direction:
            for b in barrier_direction:
                res.append((m, b))

        return res

    # def evaluate(self, chess_board, my_pos, adv_pos):
        # res1 = 0
        # res2 = 0
        # step1 = self.get_step_map(chess_board, my_pos, adv_pos)
        # step2 = self.get_step_map(chess_board, adv_pos, my_pos)
        #
        # l = len(chess_board)
        # for i in range(l):
        #     for j in range(l):
        #         res1 = res1 + step1[i][j]
        #         res2 = res2 + step2[i][j]
        #
        # return res2 - res1
        # return 0

    def evaluate(self, chess_board, my_pos, adv_pos):
        # Union-Find
        father = dict()
        l = len(chess_board)
        for r in range(l):
            for c in range(l):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        for r in range(l):
            for c in range(l):
                for dir, move in enumerate(
                        moves[1:3]
                ):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(l):
            for c in range(l):
                find((r, c))

        p0_r = find(tuple(my_pos))
        p1_r = find(tuple(adv_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return 0

        if p0_score > p1_score:
            return math.inf
        elif p0_score < p1_score:
            return -math.inf
        else:
            return -1

    def get_max(self, state, depth, max_step, alpha, beta):
        chess_board, my_pos, adv_pos = state
        if depth == self.max_depth:
            return self.evaluate(chess_board, my_pos, adv_pos), None

        max_val = self.default_max_val
        best_action = None
        moves = self.get_all_moves(chess_board, my_pos, adv_pos, max_step)
        # print ("max-moves: ", len(moves))
        l = len(moves)

        for m, b in moves:
            start = time()
            action = m, b
            state_new = self.get_next_state(chess_board, my_pos, adv_pos, 0, m, b)
            if state_new is None:
                continue

            value, _ = self.get_min(state_new, depth + 1, max_step, alpha, beta)
            # c, my, adv = state_new
            # print(my, adv, value)

            if value > max_val:
                max_val = value
                best_action = action

            if value >= beta:
                return value, action

            if value > alpha:
                alpha = value

            if l * (time() - start) > 1000:
                return self.evaluate(chess_board, my_pos, adv_pos), None

        if max_val == self.default_max_val:
            return self.evaluate(chess_board, my_pos, adv_pos), None

        return max_val, best_action

    def get_min(self, state, depth, max_step, alpha, beta):
        chess_board, my_pos, adv_pos = state
        if depth == self.max_depth:
            return self.evaluate(chess_board, my_pos, adv_pos), None

        min_val = self.default_min_val
        best_action = None
        moves = self.get_all_moves(chess_board, adv_pos, my_pos, max_step)
        # print ("min-moves: ", len(moves))
        l = len(moves)

        for m, b in moves:
            start = time()
            action = m, b
            state_new = self.get_next_state(chess_board, my_pos, adv_pos, 1, m, b)
            if state_new is None:
                continue

            value, _ = self.get_max(state_new, depth + 1, max_step, alpha, beta)
            c, my, adv = state_new
            # print(my, adv, value)

            if value < min_val:
                min_val = value
                best_action = action

            if value <= alpha:
                return value, action

            if value < beta:
                beta = value

            if l * (time() - start) > 1000:
                return self.evaluate(chess_board, my_pos, adv_pos), None

        if min_val == self.default_min_val:
            return self.evaluate(chess_board, my_pos, adv_pos), None

        return min_val, best_action
