import numpy as np
import pygame



def field_to_rgb(field):
    updraft_map = (((field - np.min(field)) / (
            np.max(field) - np.min(field))) * 255).astype(np.uint8)
    size = updraft_map.shape[1::-1]
    updraft_map = np.repeat(updraft_map.reshape(size[1], size[0], 1), 3, axis=2)
    return updraft_map


def subsurface_prep(updraft_map, x, y, resolution, screen_width=500, screen_height=500) -> np.ndarray:
    y1 = int((x - screen_width / 2) / resolution)
    y2 = int((x + screen_width / 2) / resolution)
    x1 = int((y - screen_height / 2) / resolution)
    x2 = int((y + screen_height / 2) / resolution)

    blue = np.array([89, 216, 255], dtype=np.uint8)
    localmap = np.repeat(blue, int(screen_width/resolution) * int(screen_height/resolution)).reshape((int(screen_width/resolution), int(screen_height/resolution), 3))
    j1 = 0
    j2 = None
    i1 = 0
    i2 = None
    if y1 < 0:
        j1 = int(screen_height / resolution) - y2
        j2 = None
        y1 = 0

    elif y2 > updraft_map.shape[1]:
        j1 = 0
        j2 = int(screen_height / resolution) - x2 + updraft_map.shape[0]
        y2 = None

    if y2 <= 0 or y1 >= updraft_map.shape[1]:
        return localmap

    if x1 < 0:
        i1 = int(screen_width / resolution) - x2
        i2 = None
        x1 = 0

    elif x2 > updraft_map.shape[0]:
        i1 = 0
        i2 = int(screen_width / resolution) - x2 + updraft_map.shape[0]
        x2 = None

    if x2 <= 0 or x1 >= updraft_map.shape[0]:
        return localmap
    rows, columns = localmap[i1:i2, j1:j2, :].shape[0:2]
    localmap[i1:i2, j1:j2, :] = updraft_map[x1:x1+rows, y1:y1+columns, :].astype(np.uint8)
    return localmap

def subsurface(updraft_map: np.ndarray, x: float, y: float, resolution: int, screen_width: int = 500, screen_height: int = 500) -> pygame.Surface:
    localmap = subsurface_prep(updraft_map, x, y, resolution, screen_width, screen_height)
    size = localmap.shape[1::-1]
    surface = pygame.image.frombuffer(localmap.flatten(), size, 'RGB')
    surface = pygame.transform.scale_by(surface, resolution)
    return surface


def draw_plane(plane, surface, font, wind):
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
    assert (np.isnan([plane.objective[0] - plane.x, plane.objective[1] - plane.y]).any() != True)
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
    updraft = wind.updraft(plane.x, plane.y, plane.z)
    sink = plane.sink()
    text2 = f'netto vario: {round(updraft + sink, 1)} m/s dV : {round(plane.dV,2)} m/s'
    rendered_text2 = font.render(text2, True, 'red')
    surface.blit(rendered_text2, (10, 30))
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
