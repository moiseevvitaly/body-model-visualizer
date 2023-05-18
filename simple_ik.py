import time
import torch
import numpy as np
from tqdm import tqdm
from smplx import SMPL
from loguru import logger
from measurement import VERTICES_IDX_BY_MEASUREMENT, POINTS_VARS_BY_MEASUREMENT, MEASUREMENT_NAMES
import copy

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        val = func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f'"{func.__name__}" fn took {end - start:.3f} seconds.')
        return val

    return wrapper

def make_measurements(model, betas):
    pose_params = torch.zeros(1,69)
    pose_params[0][47] = 5.6
    pose_params[0][50] = -5.6
    model_output = model(betas=betas,body_pose=pose_params)
    vertices = model_output.vertices[0]
    
    measurements = torch.zeros(len(MEASUREMENT_NAMES))
    
    for i, measurement_name in enumerate(MEASUREMENT_NAMES):
        #print(measurement_name)
        for path in VERTICES_IDX_BY_MEASUREMENT[measurement_name][:1]:
            path_length = torch.zeros(1)
            for j in range(len(path)):
                #print(path[j])
                path_length += torch.sum(torch.square(vertices[path[j]] - vertices[path[j-1]]))
            #print(path_length)
            measurements[i] += torch.sqrt(path_length)[0]
        #measurements[i] /= len(VERTICES_IDX_BY_MEASUREMENT[measurement_name])
    
    return measurements


#чо мы хотим:
#опция "увеличить параметр x"
#опция "reset parameters"
#чо делаем:
#"увеличить параметр x":
#- берем исходные параметры модели
#- вычисляем желаемые
#- запускаем оптимизацию (ищем бета параметры новые!)
#- пишем лосс
#- генерим модель с найденными бетами и в исходной позе

@timeit
def measurements_ik_solver(model, target, init_betas, device='cpu', max_iter=20,
                               mse_threshold=1e-8):
    print("init betas are: {}".format(init_betas))
    optim_betas = copy.deepcopy(init_betas)
    optim_betas = optim_betas.reshape(-1).unsqueeze(0).to(device)
    optim_betas = optim_betas.requires_grad_(True)

    optimizer = torch.optim.Adam([optim_betas], lr=0.1)
    last_mse = 0
    
    default_measurements = make_measurements(model, torch.zeros(1,10))
    current_measurements = make_measurements(model, init_betas)

    for i in range(max_iter):
        print("measurements", torch.mean(torch.square((make_measurements(model, optim_betas) - target) / current_measurements)))
        print("betas", torch.mean(torch.square(optim_betas - init_betas)))
        mse = torch.mean(torch.square((make_measurements(model, optim_betas) - target) / default_measurements)) + 0.01 * torch.mean(torch.square(optim_betas - init_betas))
        print(mse)
        if abs(mse - last_mse) < mse_threshold:
            return copy.deepcopy(torch.tensor(optim_betas.detach().numpy()))
        optimizer.zero_grad()
        mse.backward(retain_graph=True)
        optimizer.step()
        last_mse = mse

    print(f'IK final loss {last_mse.item():.3f}')
    return copy.deepcopy(torch.tensor(optim_betas.detach().numpy()))
@timeit
def simple_ik_solver(model, target, init=None, global_orient=None, device='cpu', max_iter=20,
                     mse_threshold=1e-8, transl=torch.zeros(1, 3), betas=None):
    if init is None:
        init_pose = torch.zeros(1, 69, requires_grad=True).to(device)
    else:
        init_pose = init.reshape(-1).unsqueeze(0).to(device)
        init_pose = init_pose.requires_grad_(True)
    if global_orient is None:
        global_orient_pose = torch.zeros(1, 3, requires_grad=True).to(device)
    else:
        global_orient_pose = global_orient.reshape(-1).unsqueeze(0).to(device)
        global_orient_pose = global_orient_pose.requires_grad_(True)

    optimizer = torch.optim.Adam([init_pose, global_orient_pose], lr=0.1)
    last_mse = 0
    for i in range(max_iter):

        mse = torch.mean(torch.square((
                model(
                    body_pose=init_pose,
                    global_orient=global_orient_pose,
                    betas=betas,
                    transl=transl,
                ).joints[0,:24] - target)))
        # print(i, mse.item())
        if abs(mse - last_mse) < mse_threshold:
            return init_pose, global_orient_pose
        optimizer.zero_grad()
        mse.backward(retain_graph=True)
        optimizer.step()
        last_mse = mse
    logger.info(f'IK final loss {last_mse.item():.3f}')
    return init_pose, global_orient_pose

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SMPL(f'data/body_models/smpl').float()
    joints = model().joints[0,:24]
    # joints[22] = joints[22] + 0.3
    # joints[20] = joints[20] + 0.3

    target_joints = joints + torch.rand_like(joints) * 0.1

    opt_params = simple_ik_solver(model, target_joints, max_iter=100)

    opt_joints = model(body_pose=opt_params).joints[0,:22]

    opt_joints = opt_joints.detach().numpy()
    target_joints = target_joints.detach().numpy()

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    RADIUS = 1.0
    xroot, yroot, zroot = target_joints[0, 0], target_joints[0, 1], target_joints[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    ax.scatter(target_joints[:, 0], target_joints[:, 1], target_joints[:, 2], c='b', marker='x')
    ax.scatter(opt_joints[:, 0], opt_joints[:, 1], opt_joints[:, 2], c='r', marker='o')
    plt.show()

