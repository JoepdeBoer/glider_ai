from datetime import datetime

from custom_gym.environement import Windfield

if __name__ == "__main__":
    # OEW, Water, V1,      S1, V2,      S2,   V3,      S3,  Surface
    # 350, 182, 103.77, -0.72, 155.65, -1.55, 190.24, -3.1, 10.58
    # VNE = 250 km/h = 69.44 m/s
    # Vstall = 66 km/h = 18.333

    # Plane initialisation parameters
    x0 = 100
    y0 = 100
    z0 = 1500
    V0 = 30
    theta0 = 0
    s1 = -0.72
    V1 = 28.825
    s2 = -1.55
    V2 = 43.24
    s3 = -3.1
    V3 = 52.84
    vstall = 18.33
    vne = 69.44

    #Environement initiaisation parameters
    size = 100000
    resolution = 10
    thermalheight_avg = 1500
    thermalheight_std = 50
    thermalstrenght_avg = 10
    thermalstrenght_std = 0.5
    thermalrad_avg = 200
    thermalrad_std = 30
    name  = f'Wind_{datetime.today().strftime("%m-%d-%H-%M")}'

    # Create Windfield
    for i in range(30):
        Wind = Windfield(size, resolution, thermalheight_avg, thermalheight_std, thermalstrenght_avg, thermalstrenght_std, thermalrad_avg, thermalrad_std)
        Wind.create_thermals()
        Wind.generate_field()
        name = f'random_field_{i}'
        Wind.save_field(name)

    # # Create Agent
    # Discus = Agent(100, 100, 1500, 30, 0, -0.72, 28.825, -1.55, 43.24, -3.1, 52.84, 18.33, 69.44, "Discus")
    #
    # for time in range(6000):
    #     Discus.move(Wind)
    #     if time < 50:
    #         Discus.take_action(0.0, -0.5)
    #     else:
    #         Discus.take_action(0.0, 0.1)
    #
    # Discus.plot_history()

