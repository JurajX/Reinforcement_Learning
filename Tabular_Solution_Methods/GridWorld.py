import numpy as np
import pygame
from pygame.locals import QUIT

# Globally defined colors
#               R    G    B    A
TRANSPARENT = (  0,   0,   0,   0) # transparent color
BLACK       = (  0,   0,   0)      # black color
STATE_COLOR = (255, 255, 255)      # white color
GOAL_COLOR  = ( 21, 185,   0)      # green color
WALL_COLOR  = (  0,  59, 174)      # blue color
CLIFF_COLOR = (255,  43,   0)      # red color
TRAP_COLOR  = (255, 120,   0)      # pink color

#============================================= World ==============================================#

class Worlds(object):
    """
    This class contains basic worlds for GridWorld class. For now there are three worlds:
    cliff, mirror, and maze. Use get('world_name') to obtain one of the worlds.

    Possible worlds: cliff, mirror, maze, empty, and track.

    Constructing new worlds:
    d - default, agent can access,
    s - start, agent can access,
    t - trap, agent can access, results in a special reward
    w - wall, agent can't access, actions leading to this state do not change agent's position
    c - cliff, agent can access, accessing this state puts the agent at one of the start states
    g - goal, agent can access, results in termination of the episode.
    """
    def __init__(self):
        self.cliff = np.array(
                [[ 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd'],
                 [ 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd'],
                 [ 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd'],
                 [ 's', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'g']])

        self.mirror = np.array([
                ['t', 't', 'w', 'w', 'w', 'w', 'w', 'd', 'd', 'd', 'd', 't', 'd', 'd'],
                ['t', 's', 'w', 't', 't', 't', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd'],
                ['t', 'd', 'd', 'd', 'd', 'd', 'd', 't', 't', 't', 'w', 'w', 'w', 'd'],
                ['w', 'w', 'w', 't', 't', 't', 't', 'c', 't', 't', 'w', 'g', 't', 'd'],
                ['g', 'g', 'g', 'w', 't', 't', 't', 'c', 'g', 'c', 'w', 'g', 'd', 'd'],
                ['g', 'g', 'g', 'w', 't', 't', 't', 'c', 'g', 'c', 'w', 'g', 'd', 'd'],
                ['w', 'w', 'w', 't', 't', 't', 't', 'c', 't', 't', 'w', 'g', 't', 'd'],
                ['t', 'd', 'd', 'd', 'd', 'd', 'd', 't', 't', 't', 'w', 'w', 'w', 'd'],
                ['t', 's', 'w', 't', 't', 't', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd'],
                ['t', 't', 'w', 'w', 'w', 'w', 'w', 'd', 'd', 'd', 'd', 't', 'd', 'd'],
                ])

        self.maze = np.array([
                ['g','w','c','c','c','c','d','d','d','d','w','g','d','d','d','d','d','d','d','d','c'],
                ['d','d','d','d','d','d','d','d','w','d','w','w','w','w','w','w','w','w','w','d','d'],
                ['d','d','d','d','d','d','d','d','w','d','w','d','d','d','w','d','d','d','w','w','t'],
                ['t','w','c','c','c','c','d','d','w','d','w','d','w','d','t','d','w','d','d','d','d'],
                ['c','w','w','w','w','w','w','w','w','d','w','d','w','d','d','d','w','d','d','d','d'],
                ['t','d','d','w','d','d','d','d','w','d','w','d','w','w','w','w','w','w','w','d','d'],
                ['t','t','d','w','d','w','w','d','w','d','w','d','w','d','d','d','d','d','d','d','d'],
                ['t','w','d','w','d','w','d','d','w','d','w','d','w','t','w','w','w','w','t','d','t'],
                ['t','w','d','t','d','w','d','w','w','s','w','s','c','d','d','d','c','d','d','d','t'],
                ['g','w','d','d','d','w','d','d','s','c','w','c','s','d','t','d','d','d','w','t','c'],
                ['w','w','w','w','w','w','w','w','w','w','w','w','w','w','w','w','w','w','w','w','c'],
                ['t','t','t','t','t','t','t','s','d','d','w','s','d','t','d','t','d','d','d','d','c'],
                ['t','w','w','w','w','w','w','w','w','c','w','d','d','w','w','w','d','d','d','d','t'],
                ['t','t','t','t','t','t','t','t','t','t','w','d','d','d','d','w','w','d','d','d','d'],
                ['w','w','w','w','w','w','w','w','w','t','w','d','w','w','d','d','w','w','w','d','w'],
                ['g','t','t','t','t','t','t','t','t','t','w','c','w','t','d','d','d','d','d','d','d'],
                ['w','w','w','w','w','w','w','w','w','w','w','w','w','w','w','d','d','d','d','d','d'],
                ['g','d','d','d','d','d','d','d','d','d','d','d','d','d','w','w','w','w','w','t','d'],
                ['c','c','t','d','d','d','d','d','d','d','w','t','d','d','d','d','d','d','d','t','d'],
                ['c','c','t','d','d','d','d','d','d','d','w','t','d','d','d','d','d','d','d','d','d'],
                ['g','d','d','d','d','d','d','d','d','d','d','d','d','d','w','w','w','w','w','d','d'],
                ])

        self.track = np.array([
                ['s','d','d','d','d','d','d','d','d','d','d','d','d','d','d','d','d','d','d','g'],
                ['s','d','d','d','d','d','d','d','d','d','d','d','d','d','d','d','d','d','d','g'],
                ['s','d','d','d','d','d','d','d','d','d','d','d','d','d','d','d','d','d','d','g'],
                ['s','d','d','d','d','d','d','d','d','d','d','d','d','d','d','d','d','d','d','g'],
                ['s','d','d','d','d','d','d','d','d','d','d','d','d','d','d','d','d','d','d','g'],
                ['s','d','d','d','d','d','d','d','d','d','d','d','d','d','d','d','d','d','d','g'],
                ['s','d','d','d','d','d','d','d','d','d','d','d','d','d','d','d','d','d','d','g'],
                ['s','d','d','d','d','d','d','d','d','d','d','d','d','d','d','d','d','d','d','g'],
                ])

    def get(self, world):
        if   world == 'cliff':
            return self.cliff
        elif world == 'mirror':
            return self.mirror
        elif world == 'maze':
            return self.maze
        elif world == 'track':
            return self.track

#=========================================== GridWorld ============================================#

class GridWorld(object):
    """
    GridWorld class with GUI.
    name:    Name of the environment.
    world:   One of the state spaces returned from Worlds class.
    GUI:     Default False. If set to True, a pygame window showing agent moves is displayed.
    rewards: 5-element 1D array containing the goal, cliff, trap rewards, the step cost, and
             penalty for hitting the wall (to teach agent not to do that - causes slowing).

    Member functions:
             reset:  Reset the environment, i.e. move the agent to one of the start states,
                     and sets the attribute done to False.
              step:  Perform selected action, must be one of the following characters: N, S, W, E.
                     It returns agent's new coordinates, reward, done, and info.
            render:  Display the GUI, to see all moves it needs to be called after every step.
                     If useGUI is False a message is printed. If a action-value function is
                     provided, the method displays agent's preferred actions, and state values
                     in shades of green.
        set_useGUI:  Set the boolean attribute useGUI.
         get_shape:  Return the dimension of state space as tuple.
    get_agent_coos:  Return agent's coordinates.
           is_done:  Return boolean variable indicating if the episode is over.
    """
    def __init__(self, world, rewards=[100,-50,-10,-1,-5], useGUI=True, name=None):
        self.name = name
        self.world = self._add_walls(world.T)

        # Rewards & Step Cost variables
        self.goal_reward = rewards[0]
        self.cliff_reward = rewards[1]
        self.trap_reward = rewards[2]
        self.step_cost = rewards[3]
        self.hit_wall = rewards[4]

        # Coordinates
        self.start_state_coos = self._find_start_state_coos()
        self.agent_coos = None

        # GUI
        self.GUI = GW_GUI(name=self.name, world=self.world)
        self.useGUI = useGUI

        # Episode ended
        self.done = True

    #----------------------

    def _add_walls(self, world):
        bordered_space = np.full( np.add(world.shape, (2,2)), 'w')
        bordered_space[1:-1, 1:-1] = world
        return bordered_space


    def _find_start_state_coos(self):
        coos = []
        start_indices = np.where(self.world=='s')
        start_state_number = len(start_indices[0])
        for i in range(start_state_number):
            coos.append( (start_indices[0][i], start_indices[1][i]) )

        # In case there is no start state, print warning and make all default states start states.
        if len(coos) == 0:
            #raise UserWarning
            print('UserWarning: world does not contain any start states; \
                   all default states used as start states.')
            start_indices = np.where(self.world=='d')
            start_state_number = len(start_indices[0])
            for i in range(start_state_number):
                coos.append( (start_indices[0][i], start_indices[1][i]) )
        # In case there are neither start nor default states raise ValueError.
        if len(coos) == 0:
            raise ValueError('world does not contain neither start not default states.')

        return coos

    #===========================================================================

    def reset(self):
        """
        Reset the environment, i.e. place the agent on one of the start states
        and set self.done to False.
        """
        self.agent_coos = self._pick_random_start_state()
        self.done = False

        # Reset GUI
        self.GUI.reset(self.agent_coos)

    #----------------------

    def _pick_random_start_state(self):
        number = np.random.randint(len(self.start_state_coos))
        return self.start_state_coos[number]

    #===========================================================================

    def step(self, action):
        """
        Perform the given action and return the new agent's coordinates, reward, if done, and info.
        """
        # new_agent_coos converted to tuple so that it can be used as an array index.
        if   action == 'N':
            new_agent_coos = np.add( self.agent_coos, ( 0,-1) )
            new_agent_coos = tuple( new_agent_coos )
        elif action == 'S':
            new_agent_coos = np.add( self.agent_coos, ( 0, 1) )
            new_agent_coos = tuple( new_agent_coos )
        elif action == 'W':
            new_agent_coos = np.add( self.agent_coos, (-1, 0) )
            new_agent_coos = tuple( new_agent_coos )
        elif action == 'E':
            new_agent_coos = np.add( self.agent_coos, ( 1, 0) )
            new_agent_coos = tuple( new_agent_coos )

        if   self.world[new_agent_coos] == 'd' or \
             self.world[new_agent_coos] == 's':
            self.agent_coos = new_agent_coos
            reward = self.step_cost
            self.done = False
            info = 'Agent moved to a new location.'
        elif self.world[new_agent_coos] == 't':
            self.agent_coos = new_agent_coos
            reward = self.trap_reward
            self.done = False
            info = 'Agent stepped into a trap.'
        elif self.world[new_agent_coos] == 'w':
            reward = self.hit_wall
            self.done = False
            info = 'Agent hit the wall, no change in the position.'
        elif  self.world[new_agent_coos] == 'c':
            # If the agent steps into cliff state, display it; then display a new start state.
            if self.useGUI == True:
                self.agent_coos = new_agent_coos
                self.render()
                self.agent_coos = self._pick_random_start_state()
                self.render()
            else:
                self.agent_coos = self._pick_random_start_state()
            reward = self.cliff_reward
            self.done = False
            info = 'Agent fell of the cliff, and was put at one of the starting positions.'
        elif  self.world[new_agent_coos] == 'g':
            self.agent_coos = new_agent_coos
            # This additional render is required due to a weird handling by pygame.
            # The last move to the goal state would not be otherwise displayed.
            if self.useGUI == True:
                self.render()
            reward = self.goal_reward
            self.done = True
            info = 'Agent found a goal state. Congratulations!'

        return self.agent_coos, reward, self.done, info

    #===========================================================================

    def render(self, actionVF=None):
        """
        Display the GridWorld on GUI.
        """
        if self.useGUI == True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
            self.GUI.render(self.agent_coos, actionVF)
        #else:
        #    print('useGUI is set to False; you can change it with set_useGUI member function.')
    #===========================================================================

    def get_shape(self):
        """
        Return a tuple of state space dimension.
        """
        return self.world.shape

    #----------------------

    def get_agent_coos(self):
        """
        Return agent coordinates (tuple).
        """
        return self.agent_coos

    #----------------------

    def is_done(self):
        """
        Return if the episode is over.
        """
        return self.done

    #----------------------

    def set_useGUI(self, useGUI):
        """
        Sets if a GUI should be used.
        """
        self.useGUI = useGUI

#=================================== GW_GUI: GUI for GridWorld ====================================#
#An instance of this class is created in the GridWorld class and takes care of all GUI machinery.

class GW_GUI(object):

    def __init__(self, name, world):
        pygame.init()

        self.name = name
        self.world = world

        # keep the order: WIN_SIZE needs TILE_SIZE
        self.TILE_SIZE = self._get_tile_size()
        self.WIN_SIZE = self._get_win_size()

        self.display_surface = pygame.display.set_mode(self.WIN_SIZE)
        self.display_surface.fill(BLACK)
        pygame.display.set_caption('GridWorld - ' + self.name)

        self.bg_surface = self._get_bg_surface()

        self.all_sgt_tiles = pygame.sprite.Group()
        self._add_sgt_tiles()

        self.agent = Agent(self.TILE_SIZE, (0,0))
        self.all_agents = pygame.sprite.Group()
        self.all_agents.add(self.agent)

        self.all_VF_tiles = pygame.sprite.Group()
        self._create_VF_tiles()

    #----------------------

    def _get_tile_size(self):
        state_x = 1280 // self.world.shape[0]
        state_y =  720 // self.world.shape[1]
        tile_size = min(state_x, state_y)
        if tile_size > 45:
            tile_size = 45
        return tile_size

    def _get_win_size(self):
        return (self.world.shape[0]*self.TILE_SIZE,
                self.world.shape[1]*self.TILE_SIZE)

    def _get_bg_surface(self):
        bg_surface = pygame.Surface(self.WIN_SIZE).convert_alpha()
        bg_surface.fill(BLACK)
        for i in range(self.world.shape[0]):
            for j in range(self.world.shape[1]):
                tile_type = self.world[i,j]
                tile = self._build_tile(tile_type)
                tile_rect = tile.get_rect()
                tile_rect.topleft = (i*self.TILE_SIZE, j*self.TILE_SIZE)
                bg_surface.blit(tile, tile_rect)
        return bg_surface

    def _build_tile(self, tile_type):
        tile = pygame.Surface( (self.TILE_SIZE,self.TILE_SIZE) ).convert_alpha()
        if   tile_type in 'dst':
            tile.fill(STATE_COLOR)
        elif tile_type == 'g':
            tile.fill(GOAL_COLOR)
        elif tile_type == 'w':
            tile.fill(WALL_COLOR)
        elif tile_type == 'c':
            tile.fill(CLIFF_COLOR)
        return tile

    def _add_sgt_tiles(self):
        for i in range(self.world.shape[0]):
            for j in range(self.world.shape[1]):
                tile_type = self.world[i,j]
                if tile_type in 'sgt':
                    tile = SGTtile(self.TILE_SIZE, (i,j), tile_type)
                    self.all_sgt_tiles.add(tile)

    def _create_VF_tiles(self):
        for i in range(self.world.shape[0]):
            for j in range(self.world.shape[1]):
                tile_type = self.world[i,j]
                if tile_type in 'dst':
                    vf_tile = VF_Tile(self.TILE_SIZE, (i,j))
                    self.all_VF_tiles.add(vf_tile)

    #===========================================================================

    def reset(self, agent_coos):
        """
        Reset agent's coordinates.
        """
        self.agent.coos = agent_coos

    #===========================================================================

    def render(self, agent_coos, actionVF=None):
        """
        Display agent and value function on GUI.
        """
        self.agent.coos = agent_coos
        self.all_agents.update()
        if actionVF is not None:
            opacity = self._calculate_opacity(actionVF)
            self.all_VF_tiles.update(opacity, actionVF)

        self.display_surface.blit(self.bg_surface, (0,0))
        self.all_VF_tiles.draw(self.display_surface)
        self.all_agents.draw(self.display_surface)
        self.all_sgt_tiles.draw(self.display_surface)

        pygame.display.flip()

    #----------------------

    def _calculate_opacity(self, actionVF):
        stateVF = np.sum(actionVF, axis=2)
        # Get maximum element and minimum element of state_VF.
        maximum = np.amax(stateVF)
        minimum = np.amin(stateVF)
        # Set step for opacity, i.e. range of opacity divided by range of state_VF.
        if maximum == minimum:
            step = 0
        else:
            step = 255.0/(maximum - minimum)
        # First off-set the state_VF, so that the lowest value is 0. Now, multiply
        # the off-set state_VF with step, this will give 0 opacity for minimum of
        # state_VF and 255 for maximum of state_VF (approximately). Finally, cast
        # the elements to integer values.
        opacity = (stateVF - minimum)*step
        opacity = opacity.astype('int64')
        return opacity

#===================================== Sprites used in GW_GUI =====================================#

class Agent(pygame.sprite.Sprite):
    """
    Creates an agent.
    Parrent class pygame.sprite.Sprite.
    """
    def __init__(self, tile_size, agent_coos):
        pygame.sprite.Sprite.__init__(self)
        self.TILE_SIZE = tile_size
        self.coos = agent_coos
        self.image = pygame.Surface( (self.TILE_SIZE, self.TILE_SIZE) ).convert_alpha()
        self.image.fill(TRANSPARENT)
        self.rect = self.image.get_rect()
        pygame.draw.circle(self.image, BLACK, self.rect.center,
                           self.TILE_SIZE//3 , self.TILE_SIZE//5)

    def update(self):
        self.rect.topleft = tuple(np.multiply(self.coos, self.TILE_SIZE))

#===========================================================================

class VF_Tile(pygame.sprite.Sprite):
    """
    Creates a value function tile to display value of a state and agent's preferred actions.
    Parrent class pygame.sprite.Sprite.
    """
    def __init__(self, tile_size, tile_coos):
        pygame.sprite.Sprite.__init__(self)
        self.TILE_SIZE = tile_size
        self.coos = tile_coos
        self.color = pygame.Color(21,185,0,0)
        self.sVF = pygame.Surface( (self.TILE_SIZE, self.TILE_SIZE) ).convert_alpha()
        self.sVF.fill(self.color)
        self.aVF = pygame.Surface( (self.TILE_SIZE, self.TILE_SIZE) ).convert_alpha()
        self.aVF.fill(TRANSPARENT)
        self.image = pygame.Surface( (self.TILE_SIZE, self.TILE_SIZE) ).convert_alpha()
        self.image.fill(TRANSPARENT)
        self.rect = self.image.get_rect()
        self.rect.topleft = tuple(np.multiply(self.coos, self.TILE_SIZE))

    def update(self, opacity, actionVF):
        self._update_sVF(opacity)
        self._update_aVF(actionVF)
        self.image.fill(TRANSPARENT)
        self.image.blit(self.sVF, self.sVF.get_rect())
        self.image.blit(self.aVF, self.aVF.get_rect())

    def _update_sVF(self, opacity):
        alpha = int(opacity[self.coos])
        self.color.a = alpha
        self.sVF.fill(self.color)

    def _update_aVF(self, actionVF):
        values = actionVF[self.coos]
        minimum = np.amin(values)
        values = values - minimum
        if np.sum(values) != 0:
            values = (values/np.sum(values))*(self.TILE_SIZE/3)
        else:
            values = np.ones(values.shape)*2
        dic = {'N':values[0], 'S':values[1], 'W':values[2], 'E':values[3]}
        self.aVF.fill(TRANSPARENT)
        x,y = self.aVF.get_rect().center
        pygame.draw.line(self.aVF, BLACK, (x,y-int(dic['N'])), (x,y+int(dic['S'])), 1)
        pygame.draw.line(self.aVF, BLACK, (x-int(dic['W']),y), (x+int(dic['E']),y), 1)

#===========================================================================

class SGTtile(pygame.sprite.Sprite):
    """
    Creates one of either start, goal, or trap tiles.
    Parrent class pygame.sprite.Sprite.
    """
    def __init__(self, tile_size, tile_coos, tile_type):
        pygame.sprite.Sprite.__init__(self)
        self.coos = tile_coos
        if tile_type in 'sg':
            self.image = self._build_sg_sprite(tile_size, tile_type)
        elif tile_type == 't':
            self.image = self._build_t_sprite(tile_size)
        self.rect = self.image.get_rect()
        self.rect.topleft = tuple(np.multiply(self.coos, tile_size))

    #----------------------

    def _build_sg_sprite(self, tile_size, tile_type):
        if tile_type == 'g':
            text = 'G'
            color = BLACK
        elif tile_type == 's':
            text = 'S'
            color = BLACK
        font_size = int(0.8*tile_size)
        font = pygame.font.SysFont(None, font_size)
        font.set_bold(True)
        tile = pygame.Surface( (tile_size,tile_size) ).convert_alpha()
        tile.fill(TRANSPARENT)
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        text_rect.topleft = tile.get_rect().topleft
        tile.blit(text_surface, text_rect)
        return tile

    def _build_t_sprite(self, tile_size):
        tile = pygame.Surface( (tile_size,tile_size) ).convert_alpha()
        tile.fill(TRANSPARENT)
        pygame.draw.rect(tile, TRAP_COLOR, tile.get_rect(), tile_size//5)
        return tile
