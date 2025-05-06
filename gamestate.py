from const import BACKGROUND_Y, INITIAL_GAME_SPEED


class GameState:
    obstacles = []
    dinosaurs = []
    gen_pool = []
    nets = []
    points = 0
    game_speed = INITIAL_GAME_SPEED
    x_pos_bg = 0
    y_pos_bg = BACKGROUND_Y
    spawn_cooldown = 2000
    last_spawn_time = 0
    population = None
    current_generation = 0
    score_list=[]

    @staticmethod
    def reset():        
        GameState.obstacles = []
        GameState.dinosaurs = []
        GameState.gen_pool = []
        GameState.nets = []
        GameState.points = 0
        GameState.game_speed = INITIAL_GAME_SPEED
        GameState.x_pos_bg = 0
        GameState.y_pos_bg = BACKGROUND_Y
        


    @staticmethod
    def restart():      
        GameState.obstacles = []
        GameState.dinosaurs = []
        GameState.gen_pool = []
        GameState.nets = []
        GameState.score_list=[]
        GameState.points = 0
        GameState.game_speed = INITIAL_GAME_SPEED
        GameState.x_pos_bg = 0
        GameState.y_pos_bg = BACKGROUND_Y
        GameState.spawn_cooldown = 2000
        GameState.last_spawn_time = 0
        GameState.population = None
        GameState.current_generation=0