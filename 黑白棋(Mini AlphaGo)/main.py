import math
from copy import deepcopy
import time
import random
class Node:
    def __init__(self,parent,color,board,action):
        self.parent = parent
        self.color = color
        self.board = board
        self.action = action#父节点做的action到底该节点
        self.children = []
        self.visited = 0
        self.unvisited_actions = list(board.get_legal_actions(self.color))

        self.isover = self.gameover(board) # 是否结束了

        self.reward = {'X': 0, 'O': 0}#回报值

    def add_child(self,node):
        self.children.append(node)
        self.unvisited_actions.remove(node.action)

    def is_full_expended(self):
        return len(self.unvisited_actions) == 0

    def calUCB1(self,C, color):
        if self.visited == 0:
            return float('inf')
        else:
            return self.reward[color] / self.visited + C * math.sqrt(2 * math.log(self.parent.visited) / self.visited)

    def gameover(self, board):
        l1 = list(board.get_legal_actions('X'))
        l2 = list(board.get_legal_actions('O'))
        return len(l1)==0 and len(l2)==0



class MCT:
    def __init__(self,board,color):
        self.board =deepcopy(board)
        self.color = color
        self.root = Node(None,color,board,None)
        self.C = 1.44
        self.max_times=50

    def is_game_over(self,board):
        return len(list(board.get_legal_actions('X')))==0 and len(list(board.get_legal_actions('O')))==0

    def best_child(self, node, C, color):
        # 对每个子节点调用一次计算UCBValue,选出最优的child
        bestVal = float('-inf')
        bestChild = None
        for child in node.children:
            val = child.calUCB1(C, color)
            if val > bestVal:
                bestVal = val
                bestChild = child
        return bestChild

    def search(self,C):

        node = self.root

        if 'A1' in node.state.get_legal_actions(self.color):
            return 'A1'
        if 'A8' in node.state.get_legal_actions(self.color):
            return 'A8'
        if 'H8' in node.state.get_legal_actions(self.color):
            return 'H8'
        if 'H1' in node.state.get_legal_actions(self.color):
            return 'H1'

        #只有一种选择的情况
        if len(node.unvisited_actions)==1:
            return node.unvisited_actions[0]
        #计时
        begin_time=time.time()
        while time.time()-begin_time<self.max_times:
            next_node=self.select_node(node)#选择
            result=self.Simulation(next_node)#模拟
            self.backpropagation(next_node,result)#回溯

        return self.best_child(node,0,self.color).action


    def select_node(self,node):

        selectednode = node
        while not selectednode.isover:
            if not selectednode.is_full_expended():
                if selectednode.visited<10:
                    return selectednode
                return self.expand(selectednode)
            else:
                selectednode = self.best_child(selectednode, self.C, selectednode.color)
                #无处可走了，如果不加判断会报错
                if len(selectednode.children)==0:
                    return selectednode

        return selectednode

    def expand(self,node):

        
            

        action = random.choice(node.unvisited_actions)
        newboard = deepcopy(node.board)
        
        if len(node.unvisited_actions)==0:
            pass
        else:
            newboard._move(action,node.color)

        if node.color=='X':
            newcolor='O'
        else:
            newcolor='X'

        new_node = Node(node,newcolor,newboard,action)
        node.add_child(new_node)
        return new_node
    

    def Simulation(self,node):
        newBoard = deepcopy(node.board)
        newColor = node.color

        #随机模拟
        while not self.is_game_over(newBoard):
            actions = list(newBoard.get_legal_actions(newColor))
            #该颜色没办法走了
            if len(actions) == 0:
                action = None
            else:
                if 'A1' in actions:
                    action = 'A1'
                elif 'A8' in actions:
                    action = 'A8'
                elif 'H1' in actions:
                    action = 'H1'
                elif 'H8' in actions:
                    action = 'H8'
     
                else:
                    action = random.choice(actions)
            
            if action is None:
                pass
            else:
                newBoard._move(action, newColor)
            
            newColor = 'X' if newColor=='O' else 'O'

        winner, count = newBoard.get_winner()
        count /= 64
        return winner, count

    def backpropagation(self,node,result):
        newNode = node
        # 节点不为None时
        while newNode is not None:
            newNode.visited += 1
            #颜色结果回溯结果以正数形式加上，反之
            if result[0] == 0:
                newNode.reward['X'] += result[1]
                newNode.reward['O'] -= result[1]
            elif result[0] == 1:
                newNode.reward['X'] -= result[1]
                newNode.reward['O'] += result[1]

            newNode = newNode.parent

class AIPlayer:
    """
    AI 玩家
    """

    def __init__(self, color):
        """
        玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        """

        self.color = color

    def get_move(self, board):
        """
        根据当前棋盘状态获取最佳落子位置
        :param board: 棋盘
        :return: action 最佳落子位置, e.g. 'A1'
        """
        if self.color == 'X':
            player_name = '黑棋'
        else:
            player_name = '白棋'
        print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))

        # -----------------请实现你的算法代码--------------------------------------

        mcts = MCT(deepcopy(board), self.color)
        action = mcts.search(1.2)
        # ------------------------------------------------------------------------

        return action