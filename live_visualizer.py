import pygame
import numpy
import random
import time
from grid_world import GridWorld

# colors
BLACK      = (0, 0, 0)
WHITE      = (255, 255, 255)
DARK_GRAY  = (25, 25, 35)
WALL_HOVER = (90, 90, 110)
WALL       = (60, 60, 80)
GOAL       = (80, 200, 120)
AGENT      = (100, 180, 255)
GRID_LINE  = (20, 20, 20)
TEXT       = (220, 220, 220)

CELL_SIZE  = 120
PADDING    = 20
GRID_SIZE  = 5 # CHANGE THIS ONE NUMBER TO RESIZE EVERYTHING
SLIDER_MIN = 20
SLIDER_MAX = 300

WIDTH  = GRID_SIZE * CELL_SIZE + PADDING * 2
HEIGHT = GRID_SIZE * CELL_SIZE + PADDING * 2 + 110

# Q-Learning hyperparameters
alpha         = 0.1
gamma         = 0.9
epsilon       = 1.0
epsilon_decay = 0.995
epsilon_min   = 0.01
episodes      = 1000

q_table = numpy.zeros((GRID_SIZE * GRID_SIZE, 4))

def pos_to_state(pos):
    return pos[0] * GRID_SIZE + pos[1]

def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3)
    else:
        return numpy.argmax(q_table[state])
    
def get_cell_color(row, col, env):
    if (row, col) in env.walls:
        return WALL
    if (row, col) == env.goal:
        return GOAL
    if (row, col) == env.agent_pos:
        return AGENT
    
    # heatmap based on max Q-value for this state
    state = pos_to_state((row, col))
    max_q = numpy.max(q_table[state])

    # normalize roughly between -10 and +10
    normalized = (max_q + 10) / 20.0
    normalized = max(0.0, min(1.0, normalized))

    # interpolate from dark blue (cold) to bright orange (hot)
    r = int(normalized * 220)
    g = int(normalized * 100)
    b = int((1 - normalized) * 180)
    return (r, g, b)

def draw(screen, env, episode, steps, total_reward, font, small_font, epsilon, speed=30):
    screen.fill(BLACK)

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            x = PADDING + col * CELL_SIZE
            y = PADDING + row * CELL_SIZE
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

            color = get_cell_color(row, col, env)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, GRID_LINE, rect, 2)

            # cell labels
            if (row, col) in env.walls:
                label = small_font.render("WALL", True, WHITE)
            elif (row, col) == env.goal:
                label = small_font.render("GOAL", True, BLACK)
            elif (row, col) == env.agent_pos:
                label = small_font.render("AGENT", True, BLACK)
            else:
                # show the max Q value
                max_q = numpy.max(q_table[pos_to_state((row, col))])
                label = small_font.render(f"{max_q:.1f}", True, WHITE)

            lw = label.get_width()
            lh = label.get_height()
            screen.blit(label, (x + (CELL_SIZE - lw) // 2,
                                y + (CELL_SIZE - lh) // 2))
            
    # hud    
    hud = font.render(
        f"Episode: {episode}  Steps: {steps}  reward: {total_reward}  Eps: {epsilon:.3f}",
        True, TEXT
    )
    screen.blit(hud, (WIDTH // 2 - hud.get_width() // 2, HEIGHT - 80))

    speed_label = font.render(f"Speed: {speed} fps  (W: faster  S: slower)", True, TEXT)
    screen.blit(speed_label, (WIDTH // 2 - speed_label.get_width() // 2, HEIGHT - 50))
    pygame.display.flip()

def main():
    global epsilon
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Q-Learning Live Heatmap")
    font = pygame.font.SysFont("monospace", max(14, WIDTH // 35))
    clock = pygame.time.Clock()

    env = GridWorld(size=GRID_SIZE)

    env.walls = []
    editing = True
    small_font = pygame.font.SysFont("monospace", max(12, WIDTH // 40))

    while editing:
        mouse_pos = pygame.mouse.get_pos()
        hover_col = (mouse_pos[0] - PADDING) // CELL_SIZE
        hover_row = (mouse_pos[1] - PADDING) // CELL_SIZE
        hover_cell = (hover_row, hover_col)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if 0 <= hover_row <= GRID_SIZE and 0 <= hover_col < GRID_SIZE:
                    if event.button == 1: # left click - toggle wall
                        if hover_cell == env.goal or hover_cell == env.start:
                            pass # can't wall over goal or start
                        elif hover_cell in env.walls:
                            env.walls.remove(hover_cell)
                        else:
                            env.walls.append(hover_cell)

                    if event.button == 3: # right click - move goal
                        if hover_cell not in env.walls and hover_cell != env.start:
                            env.goal = hover_cell

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    editing = False

        # draw edit mode
        screen.fill(BLACK)

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                x = PADDING + col * CELL_SIZE
                y = PADDING + row * CELL_SIZE
                rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

                if (row, col) in env.walls:
                    color = WALL
                elif (row, col) == env.goal:
                    color = GOAL
                elif (row, col) == env.start:
                    color = AGENT
                elif (row, col) == hover_cell and 0 <= hover_row < GRID_SIZE and 0 <= hover_col < GRID_SIZE:
                    color = WALL_HOVER
                else:
                    color = DARK_GRAY

                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, GRID_LINE, rect, 2)

                # labels
                if (row, col) in env.walls:
                    label = small_font.render("WALL", True, WHITE)
                elif (row, col) == env.goal:
                    label = small_font.render("GOAL", True, BLACK)
                elif (row, col) == env.start:
                    label = small_font.render("START", True, BLACK)
                else:
                    label = None

                if label:
                    lw = label.get_width()
                    lh = label.get_height()
                    screen.blit(label, (x + (CELL_SIZE - lw) // 2,
                                        y + (CELL_SIZE - lh) // 2))
                    
        # instructions
        instruct = font.render("Left click: wall  |  Right click: goal  |  Space: start", True, TEXT)
        screen.blit(instruct, (WIDTH // 2 - instruct.get_width() // 2, HEIGHT - 75))

        pygame.display.flip()
        clock.tick(30)

    # reset Q-table for fresh training on the new maze
    q_table[:] = 0
    epsilon = 1.0

    train_speed = 30

    for episode in range(1, episodes + 1):
        state = pos_to_state(env.reset())
        done = False
        steps = 0
        total_reward = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w:
                        train_speed = min(SLIDER_MAX, train_speed + 20)
                    if event.key == pygame.K_s:
                        train_speed = max(SLIDER_MIN, train_speed - 20)
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                
            action = choose_action(state)
            new_pos, reward, done = env.step(action)
            new_state = pos_to_state(new_pos)

            # Bellman update
            best_future = numpy.max(q_table[new_state])
            q_table[state, action] = q_table[state, action] + alpha * (
                reward + gamma * best_future - q_table[state, action]
            )

            state = new_state
            total_reward += reward
            steps += 1

            draw(screen, env, episode, steps, total_reward, font, small_font, epsilon, train_speed)
            clock.tick(train_speed)

            if steps > 200:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # after training - watch pure exploitation
    print("Training complete. Watching trained agent...")
    for _ in range (10):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        while True:
            start = (random.randint(0, 4), random.randint(0, 4))
            if start not in env.walls and start != env.goal:
                break

        state = pos_to_state(env.reset(start=start))
        done = False
        steps = 0
        total_reward = 0

        while not done and steps < 50:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            action = numpy.argmax(q_table[state])
            new_pos, reward, done = env.step(action)
            state = pos_to_state(new_pos)
            total_reward += reward
            steps += 1

            draw(screen, env, episode, steps, total_reward, font, small_font, epsilon)
            time.sleep(0.3)

if __name__ == "__main__":
    main()