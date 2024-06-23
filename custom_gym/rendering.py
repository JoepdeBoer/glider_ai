import numpy as np
import pygame


def field_to_rgb(field):
    updraft_map = (((field - np.min(field)) / (
            np.max(field) - np.min(field))) * 255).astype(np.uint8)
    size = updraft_map.shape[1::-1]
    updraft_map = np.repeat(updraft_map.reshape(size[1], size[0], 1), 3, axis=2)
    return updraft_map


def subsurface(updraft_map, x, y, resolution, screen_width=500, screen_height=500):
    y1 = int((x - screen_width / 2) / resolution)
    y2 = int((x + screen_width / 2) / resolution)
    # x2 = int(x1 + round(screen_width/resolution))
    x1 = int((y - screen_height / 2) / resolution)
    x2 = int((y + screen_height / 2) / resolution)
    # y2 = int(y1 + round(screen_height/resolution))
    # x1 = int((x/resolution - screen_width / 2))
    # x2 = int(x1 + screen_width)
    # y1 = int((y/resolution - screen_height/2))
    # y2 = int(y1 + screen_height)
    out_of_bounds = []
    if x1 < 0:
        x1 = 0
        addleft = -x1
        out_of_bounds.append['left']
    elif x2 > updraft_map.shape[0]:
        x2 = updraft_map.shape[0] - 1
        addright = x2 - updraft_map.shape[0]
        out_of_bounds.append('right')
    if y1 < 0:
        y1 = 0
        addup = -y1
        out_of_bounds.append('up')
    elif y2 > updraft_map.shape[1]:
        y2 = updraft_map.shape[1] - 1
        adddown = y2 - updraft_map.shape[1]
        out_of_bounds.appemd('down')

    localmap = updraft_map[x1:x2, y1:y2, :]
    if out_of_bounds:
        blue = np.array([89, 216, 255])
        for i in out_of_bounds:
            match i:
                case 'up':
                    bluerow = np.tile(blue, (1, localmap.shape[1]))  # lenght of only the localmap
                    rows_above = np.repeat(bluerow, addup, axis=0)
                    localmap = np.stack((rows_above, localmap), axis=0)
                case 'down':
                    bluerow = np.tile(blue, (1, localmap.shape[1]))  # lenght of only the localmap
                    rows_below = np.repeat(bluerow, adddown, axis=0)
                    localmap = np.stack((localmap, rows_below), axis=0)

                case 'left':
                    bluecolumn = np.tile(blue, (int(round(screen_height / resolution)), 1))  # lenght of desired map
                    columns_left = np.repeat(bluecolumn, addleft, axis=1)
                    localmap = np.stack((columns_left, localmap), axis=1)

                case 'right':
                    bluecolumn = np.tile(blue, (int(round(screen_height / resolution)), 1))  # lenght of desired map
                    columns_right = np.repeat(bluecolumn, addright, axis=1)
                    localmap = np.stack((localmap, columns_right), axis=1)

    size = localmap.shape[1::-1]
    surface = pygame.image.frombuffer(localmap.flatten(), size, 'RGB')
    surface = pygame.transform.scale_by(surface, resolution)
    return surface


def draw_plane(plane, surface, font):
    width = surface.get_width()
    height = surface.get_height()

    r = 5
    v = np.array([[0, 3 * r, 0], [-r, 0, r]])
    R = np.array([[np.cos(plane.theta), -np.sin(plane.theta)], [-np.sin(plane.theta), -np.cos(plane.theta)]])
    v_rotated = np.dot(R, v)
    v_rotated[0] = v_rotated[0] + width / 2
    v_rotated[1] = v_rotated[1] + height / 2

    # draw history
    steps = 100
    skips = 5
    if plane.history:
        if len(plane.history) > steps + 1:
            xy = [(i[0]-plane.x + width/2, i[1]-plane.y + height/2) for i in plane.history[-steps::skips]]
            xy.insert(-1, (width/2, height/2))
            assert(np.isnan(xy).any() == False)
            pygame.draw.lines(surface, 'blue', False, xy)

    # draw objective and direction to it
    #R = OBJECTIVE_RADIUS
    R = 2000
    vector = np.array([plane.objective[0] - plane.x, plane.objective[1] - plane.y])
    distance = np.linalg.norm(vector)
    assert (np.isnan([plane.objective[0] - plane.x, plane.objective[1] - plane.y]).any() == False)
    pygame.draw.line(surface, 'green',
                       (width / 2, height / 2), (plane.objective[0] - plane.x, plane.objective[1] - plane.y))

    if distance - R < np.sqrt((width / 2) ** 2 + (height / 2) ** 2):
        objx = width / 2 + vector[0]
        objy = height / 2 + vector[1]
        pygame.draw.circle(surface, 'green', (objx, objy), R, 1)

    # draw plane by circle and triangle
    pygame.draw.polygon(surface, 'red', (v_rotated[:, 0], v_rotated[:, 1], v_rotated[:, 2]))
    pygame.draw.circle(surface, 'white', (width / 2, height / 2), r)

    # draw plane info
    text = f' h: {round(plane.z, 1)} m  V: {round(plane.V, 1)} m/s  DTO:{round(distance, 1)} m'
    rendered_text = font.render(text, True, 'red')
    surface.blit(rendered_text, (10, 10))
    return surface


if __name__ == '__main__':
    WIDTH, HEIGHT = 1000, 1000  # should be an even number
    WINDFIELD_resolution = 10  # m
    FPS = 24  # frames per second

    # VARIABLES
    OBJECTIVE_POS = (HEIGHT * 10, WIDTH * 10)
    OBJECTIVE_RADIUS = 100  # 1Km
    OBJECTIVE_COLOR = (255, 0, 0)

    PLANE_POS = (WIDTH / 2, HEIGHT / 2)  # with respect to the display
    PLANE_X = 1000  # m
    PLANE_Y = 1000  # m
    PLANE_ALT = 1000.  # m
    PLANE_V = 30.  # m/s
    PLANE_THETA = 0.  # rad
    PLANE_VARIO = 0.  # m/s
    PLANE_thickness = 1  # m
    PLANE_TRAIL_LENGHT = 10  # STEPS
    PLANE_TRAJECTORY_LENGHT = 5  # STEPS

    PLANE_COLOR = (0, 0, 255)

    # step
    STEP_count = 0
    MAX_STEP = 3 * 60 * 60  # 3h

    WIND = (0, 0)
    WINDFIELD = (0, 0)
    WINDFIELD_RADIUS = 1.  # Km

    pygame.init()
    pygame.display.set_caption("Gliding Simulation")
    display = pygame.display.set_mode((WIDTH, HEIGHT))
    field = np.load(f'./Windfields/random_field_0.npy')
    updraft_map = (((field - np.min(field)) / (
            np.max(field) - np.min(field))) * 255).astype(np.uint8)
    size = updraft_map.shape[1::-1]
    updraft_map = np.repeat(updraft_map.reshape(size[1], size[0], 1), 3, axis=2)
    # surface = pygame.image.frombuffer(updraft_map.flatten(), size, 'RGB')
    # pygame.transform.scale_by(surface, 8)
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        #draw
        #display.fill((100, 100, 100))
        x1 = int((PLANE_X - WIDTH / 2) * WINDFIELD_resolution)
        x2 = int(x1 + round(WIDTH / WINDFIELD_resolution))
        y1 = int((PLANE_Y - HEIGHT / 2) * WINDFIELD_resolution)
        y2 = int(y1 + round(HEIGHT / WINDFIELD_resolution))
        localmap = updraft_map[y1:y2, x1:x2, ]
        size = localmap.shape[1::-1]
        surface = pygame.image.frombuffer(localmap.flatten(), size, 'RGB')
        surface = pygame.transform.scale_by(surface, WINDFIELD_resolution)
        display.blit(surface, (0, 0))

        pygame.draw.circle(display, OBJECTIVE_COLOR, OBJECTIVE_POS, OBJECTIVE_RADIUS * 100)
        pygame.draw.circle(display, 'WHITE', PLANE_POS, PLANE_thickness * 10)
        pygame.display.update()

        clock.tick(FPS)
    pygame.quit()
