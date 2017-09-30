"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # Get the proportion of own moves vs that of the opponent
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    total_moves = own_moves + opp_moves

    # Make sure own_moves divided by total_moves is legal else return 0
    if total_moves == 0:
        return 0

    return float(own_moves/total_moves)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # Get each players legal moves
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    # Return own legal moves subtracted from opponent's legal moves
    return float(own_moves - 2*opp_moves)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # Get the proportion of own moves vs that of the opponent
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    move_vals = (own_moves - 2*opp_moves)

    # Get the location of player
    w, h = game.width / 2., game.height / 2.
    y, x = game.get_player_location(player)
    loc_vals = float((h - y)**2 + (w - x)**2)

    # Return score that combines proportion of moves and location
    return float(move_vals + loc_vals)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1,-1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move


    # Max helper function to find the best utililty for player
    def max_val(self, game, depth):

        # Check time constraint
        if self.time_left() < self.TIMER_THRESHOLD: raise SearchTimeout()

        # If a terminal state return utility value
        if depth == 1: return self.score(game,self)

        # Update and initialize values
        utility = float('-inf')
        legal_moves = game.get_legal_moves()

        # Loop though actions to determine best utility
        for action in legal_moves:
            utility = max(utility, self.min_val(game.forecast_move(action), depth-1))

        # Return best value
        return utility


    # Min helper function to find the lowest utililty for player
    def min_val(self, game, depth):

        # Check time constraint
        if self.time_left() < self.TIMER_THRESHOLD: raise SearchTimeout()

        # If a terminal state return utility value
        if depth == 1: return self.score(game,self)

        # Update and initialize values
        utility = float('inf')
        legal_moves = game.get_legal_moves()

        # Loop though actions to determine best utility
        for action in legal_moves:
            utility = min(utility, self.max_val(game.forecast_move(action), depth-1))

        # Return best value
        return utility


    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD: raise SearchTimeout()

        # Set and initialize values
        best_utility = float("-inf")
        legal_moves = game.get_legal_moves() 

        # If no legal move return default
        if not legal_moves:
            return (-1,-1)

        best_move = legal_moves[0]

        # Loop though actions
        for action in legal_moves:
            utility = self.min_val(game.forecast_move(action), depth)

            # Update best utility and best move
            if utility > best_utility:
                best_utility = utility
                best_move = action

        # Return best action
        return best_move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        # Check time constraint
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1,-1)

        # The try/except block will automatically catch the exception
        # raised when the timer is about to expire.
        try:
            # For depth 1 through infinite
            depth = 1
            while True:
                best_move = self.alphabeta(game, depth)
                depth += 1

                if self.time_left() < self.TIMER_THRESHOLD: raise SearchTimeout()

        except SearchTimeout:
            # Handle any actions required after timeout as needed
            return best_move

        # Return the best move from the last completed search iteration
        return best_move


    # Max helper function to find the best utililty for player
    def max_val(self, game, depth, alpha, beta):

        # Check time constraint
        if self.time_left() < self.TIMER_THRESHOLD: raise SearchTimeout()

        # If a terminal state return utility value
        if depth == 1: return self.score(game,self)

        # Update and initialize values
        utility = float("-inf")
        legal_moves = game.get_legal_moves()

        # Loop though actions to determine best utility
        for action in legal_moves:
            utility = max(utility, self.min_val(game.forecast_move(action), depth-1, alpha, beta))

            # If utility better than beta return utility
            if utility >= beta:
                return utility

            # Update alpha
            alpha = max(alpha, utility)

        # Return best value
        return utility


    # Min helper function to find the lowest utililty for player
    def min_val(self, game, depth, alpha, beta):

        # Check time constraint
        if self.time_left() < self.TIMER_THRESHOLD: raise SearchTimeout()

        # If a terminal state return utility value
        if depth == 1: return self.score(game,self)

        # Update and initialize values
        utility = float("inf")
        legal_moves = game.get_legal_moves()

        # Loop though actions to determine best utility
        for action in legal_moves:
            utility = min(utility, self.max_val(game.forecast_move(action), depth-1, alpha, beta))

            # If utility lower than alpha return utility
            if utility <= alpha:
                return utility

            # Update beta
            beta = min(beta, utility)

        # Return best value
        return utility


    def alphabeta(self, game, depth):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        # Check time constraint
        if self.time_left() < self.TIMER_THRESHOLD: raise SearchTimeout()

        # Setup and initialize values
        alpha = float("-inf")
        beta = float("inf")
        best_utility = float("-inf")
        legal_moves = game.get_legal_moves() 

        # If no legal moves return default
        if not legal_moves:
            return (-1,-1)

        # In case time runs out take a default legal move
        best_move = legal_moves[0]

        # Loop though actions
        for action in legal_moves:
            utility = self.min_val(game.forecast_move(action), depth, alpha, beta)
            
            # Update best utility and move
            if utility > best_utility:
                best_utility = utility
                best_move = action

                # If best utility is bigger than beta return best move
                if best_utility >= beta:
                    return best_move

            # Update alpha
            alpha = max(utility, alpha)

        # Return best action
        return best_move
