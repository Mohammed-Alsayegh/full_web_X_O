import random
import threading
from typing import List, Optional

from flask import Flask, jsonify, request, send_file, send_from_directory
import numpy as np

app = Flask(__name__)

# ===================== Game & RL =====================

class TicTacToe:
    def __init__(self):
        self.board: List[str] = [' '] * 9
        self.current_player: str = 'X'
        self.game_over: bool = False
        self.winner: Optional[str] = None

    def reset(self):
        self.board = [' '] * 9
        self.current_player = 'X'
        self.game_over = False
        self.winner = None

    def available_moves(self):
        return [i for i, c in enumerate(self.board) if c == ' ']

    def make_move(self, pos: int) -> bool:
        if 0 <= pos < 9 and self.board[pos] == ' ' and not self.game_over:
            self.board[pos] = self.current_player
            self._check_game_over()
            if not self.game_over:
                self.current_player = 'O' if self.current_player == 'X' else 'X'
            return True
        return False

    def _check_game_over(self):
        wins = [
            [0,1,2],[3,4,5],[6,7,8],
            [0,3,6],[1,4,7],[2,5,8],
            [0,4,8],[2,4,6]
        ]
        for a,b,c in wins:
            if self.board[a] != ' ' and self.board[a] == self.board[b] == self.board[c]:
                self.game_over = True
                self.winner = self.board[a]
                return
        if ' ' not in self.board:
            self.game_over = True
            self.winner = None  # draw

class QLearningAgent:
    def __init__(self, alpha=0.4, gamma=0.95, epsilon=0.2):
        self.q = {}  # (state, action) -> value
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def _key(self, state: str, action: int):
        return (state, action)

    def get_q(self, state: str, action: int):
        return self.q.get(self._key(state, action), 0.0)

    def set_q(self, state: str, action: int, value: float):
        self.q[self._key(state, action)] = value

    def choose(self, state: str, actions: List[int], explore=True):
        if explore and random.random() < self.epsilon:
            return random.choice(actions)
        # exploit
        q_vals = [self.get_q(state, a) for a in actions]
        mx = max(q_vals)
        best = [a for a, qv in zip(actions, q_vals) if qv == mx]
        return random.choice(best)

    def learn(self, s, a, r, s_next, next_actions):
        old = self.get_q(s, a)
        if s_next is None or not next_actions:
            target = r
        else:
            target = r + self.gamma * max(self.get_q(s_next, na) for na in next_actions)
        new = old + self.alpha * (target - old)
        self.set_q(s, a, new)

def board_to_state(board: List[str]) -> str:
    return ''.join(board)

def reward_for(player: str, winner: Optional[str]) -> int:
    if winner is None:
        return 0
    return 1 if winner == player else -1

def train_self_play(agentX: QLearningAgent, agentO: QLearningAgent, episodes=30000):
    game = TicTacToe()
    for _ in range(episodes):
        game.reset()
        last = {"X": (None, None), "O": (None, None)}  # (state, action)
        while not game.game_over:
            player = game.current_player
            agent  = agentX if player == 'X' else agentO
            s = board_to_state(game.board)
            actions = game.available_moves()
            a = agent.choose(s, actions, explore=True)
            game.make_move(a)

            # reward for the player who just moved (if terminal)
            if game.game_over:
                rX = reward_for('X', game.winner)
                rO = reward_for('O', game.winner)
                # update last move (this move)
                agent.learn(s, a, (rX if player=='X' else rO), None, [])
                # also give the opponent final feedback for its previous move
                opp = 'O' if player=='X' else 'X'
                opp_agent = agentO if opp=='O' else agentX
                ps, pa = last[opp]
                if ps is not None and pa is not None:
                    opp_r = (rO if opp=='O' else rX)
                    opp_agent.learn(ps, pa, opp_r, None, [])
                break
            else:
                # previous move of same player gets intermediate reward 0
                ps, pa = last[player]
                if ps is not None and pa is not None:
                    next_s = board_to_state(game.board)
                    next_actions = game.available_moves()
                    agent.learn(ps, pa, 0, next_s, next_actions)
                last[player] = (s, a)

# --- train once on startup ---
agent_X = QLearningAgent(alpha=0.4, gamma=0.95, epsilon=0.2)
agent_O = QLearningAgent(alpha=0.4, gamma=0.95, epsilon=0.2)
train_self_play(agent_X, agent_O, episodes=30000)
# turn off exploration for inference
agent_X.epsilon = 0.0
agent_O.epsilon = 0.0

# ================ Simple server-side session ================
session = {
    "game": TicTacToe(),
    "human": "X",   # default: human X
    "ai":    "O"
}
lock = threading.Lock()

def ai_move_if_needed():
    g: TicTacToe = session["game"]
    if g.game_over:
        return
    if g.current_player != session["ai"]:
        return
    agent = agent_X if session["ai"] == 'X' else agent_O
    s = board_to_state(g.board)
    a = agent.choose(s, g.available_moves(), explore=False)
    g.make_move(a)

def state_json():
    g: TicTacToe = session["game"]
    return {
        "board": g.board,
        "current": g.current_player,
        "game_over": g.game_over,
        "winner": g.winner,
        "human": session["human"],
        "ai": session["ai"]
    }

# ===================== Routes =====================

@app.route("/")
def index():
    return send_file("index.html")

@app.route("/static/index.css")
def css():
    return send_from_directory(".", "index.css", mimetype="text/css")

@app.route("/static/script.js")
def js():
    return send_from_directory(".", "script.js", mimetype="application/javascript")

@app.route("/state")
def get_state():
    with lock:
        return jsonify(state_json())

@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json(silent=True) or {}
    human = data.get("human", "X")
    human = "X" if str(human).upper() != "O" else "O"
    with lock:
        session["human"] = human
        session["ai"]    = "O" if human == "X" else "X"
        session["game"]  = TicTacToe()
        # لو الـ AI يبدأ
        if session["game"].current_player == session["ai"]:
            ai_move_if_needed()
        return jsonify(state_json())

@app.route("/move", methods=["POST"])
def move():
    data = request.get_json(silent=True) or {}
    pos = int(data.get("pos", -1))
    with lock:
        g: TicTacToe = session["game"]
        if g.game_over:
            return jsonify(state_json())
        if g.current_player != session["human"]:
            return jsonify(state_json())
        g.make_move(pos)
        if not g.game_over:
            ai_move_if_needed()
        return jsonify(state_json())

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)
