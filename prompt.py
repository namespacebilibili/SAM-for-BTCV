import torch
import random
import cfg

def msk_preprocess(msk):
    # msk: (b, c, h, w, d)
    # process_msk: (b, type, c, h, w, d)

    process_msk = torch.zeros((msk.size()[0], 13, msk.size()[1], msk.size()[2], msk.size()[3], msk.size()[4]),dtype=torch.float)
    for b in range(msk.size()[0]):
        single_msk = msk[b]
        for d in range(msk.size()[4]):
            now_msk = single_msk[:, :, :, d][0]
            for t in range(13):
                x = torch.full_like(now_msk, 1)
                y = torch.zeros_like(now_msk)
                uni_msk = torch.where(now_msk == t + 1, x, y)
                process_msk[b, t, 0, :, :, d] = uni_msk
    return process_msk

def msk_label_preprocess(msk):
    # msk: (b, c, h, w, d)
    # process_msk: (b, type, c, h, w, d)
    # label: (b, t, d, 13)

    process_msk = torch.zeros((msk.size()[0], 13, msk.size()[1], msk.size()[2], msk.size()[3], msk.size()[4]),dtype=torch.float)
    label = torch.zeros((msk.size()[0], 13, msk.size()[4], 1), dtype=torch.long)
    for b in range(msk.size()[0]):
        single_msk = msk[b]
        for d in range(msk.size()[4]):
            now_msk = single_msk[:, :, :, d][0]
            for t in range(13):
                x = torch.full_like(now_msk, 1)
                y = torch.zeros_like(now_msk)
                uni_msk = torch.where(now_msk == t + 1, x, y)
                process_msk[b, t, 0, :, :, d] = uni_msk

                if not torch.equal(torch.zeros(uni_msk.shape), uni_msk.cpu()):
                    label[b, t, d, 0] = t 
    return process_msk, label

def generate_prompt(args,msk):
    if args.use_multi:
        prompt_list = generate_multi_resize_prompt(msk, args.multi_num)
        point_labels = torch.ones((msk.size()[0], msk.size()[1], args.multi_num),dtype=torch.float)
        ables = [[] for _ in range(msk.size()[0])]
        for i in range(msk.size()[0]):
            ables[i] = [
                j
                for j in range(msk.size()[1])
                if not torch.allclose(
                    prompt_list[i][j][args.multi_num - 1],
                    torch.tensor(
                        [-1, -1], dtype=torch.float32
                    ),
                )
            ]
    elif args.use_pn:
        prompt_list, point_labels = generate_pn_resize_prompt(msk, args.multi_num)
        ables = [[] for _ in range(msk.size()[0])]
        for i in range(msk.size()[0]):
            ables[i] = [
                j
                for j in range(msk.size()[1])
                if not torch.allclose(
                    prompt_list[i][j][args.multi_num - 1],
                    torch.tensor(
                        [-1, -1], dtype=torch.float32
                    ),
                )
            ]
    elif args.use_box:
        prompt_list = generate_box_resize_prompt(msk)
        point_labels=None
        ables = [[] for _ in range(msk.size()[0])]
        for i in range(msk.size()[0]):
            ables[i] = [
                j
                for j in range(msk.size()[1])
                if not torch.allclose(
                    prompt_list[i][j][0],
                    torch.tensor(
                        [-1], dtype=torch.float32
                    ),
                )
            ]
    else:
        prompt_list = generate_resize_prompt(msk)
        point_labels = torch.ones((msk.size()[0], msk.size()[1]),dtype=torch.float)
        ables = [[] for _ in range(msk.size()[0])]
        for i in range(msk.size()[0]):
            ables[i] = [
                j
                for j in range(msk.size()[1])
                if not torch.allclose(
                    prompt_list[i][j],
                    torch.tensor(
                        [-1, -1], dtype=torch.float32
                    ),
                )
            ]

    return prompt_list, point_labels, ables


def generate_resize_prompt(msk):
    # msk: (bd, t, 1, 1024, 1024)
    # prompt_list: (bd, t, 2)
    prompt_list = torch.zeros((msk.size()[0], msk.size()[1], 2),dtype=torch.float)
    for i in range(msk.size()[0]):
        for j in range(msk.size()[1]):
            # single_msk: (1024, 1024)
            single_msk = msk[i][j][0]
            able_area = torch.nonzero(single_msk)
            if able_area.size()[0] == 0:
                point_prompt = torch.tensor([-1, -1], dtype=torch.float)
            else:
                # print(now_msk[msk_type].squeeze(0).size())
                # print(able_area.size())
                random_choice = random.randint(0, able_area.size()[0] - 1)
                point_prompt = able_area[random_choice]
                # print(point_prompt)
            prompt_list[i,j,:] = point_prompt
    return prompt_list

def generate_multi_resize_prompt(msk, multi_num):
    # msk: (bd, t, 1, 1024, 1024)
    # prompt_list: (bd, t, k, 2)
    prompt_list = torch.zeros((msk.size()[0], msk.size()[1], multi_num, 2),dtype=torch.float)
    for i in range(msk.size()[0]):
        for j in range(msk.size()[1]):
            # single_msk: (1024, 1024)
            single_msk = msk[i][j][0]
            able_area = torch.nonzero(single_msk)
            if able_area.size()[0] == 0:
                point_prompt = torch.full((multi_num, 2), -1, dtype=torch.float)
                prompt_list[i, j, :, :] = point_prompt
            else:
                # print(now_msk[msk_type].squeeze(0).size())
                # print(able_area.size())
                for k in range(multi_num):
                    random_choice = random.randint(0, able_area.size()[0] - 1)
                    prompt_list[i, j, k, :] = able_area[random_choice]
    return prompt_list

def generate_pn_resize_prompt(msk, multi_num):
    # msk: (bd, t, 1, 1024, 1024)
    # prompt_list: (bd, t, k+1, 2)
    # label_list: (bd, t, k+1)
    prompt_list = torch.zeros((msk.size()[0], msk.size()[1], multi_num+1, 2),dtype=torch.float)
    label_list = torch.ones((msk.size()[0], msk.size()[1], multi_num+1),dtype=torch.float)
    for i in range(msk.size()[0]):
        for j in range(msk.size()[1]):
            # single_msk: (1024, 1024)
            single_msk = msk[i][j][0]
            able_area = torch.nonzero(single_msk)
            disable_area = torch.nonzero(1 - single_msk)
            if able_area.size()[0] == 0:
                point_prompt = torch.full((multi_num+1, 2), -1, dtype=torch.float)
                prompt_list[i, j, :, :] = point_prompt
            else:
                for k in range(multi_num):
                    random_choice = random.randint(0, able_area.size()[0] - 1)
                    prompt_list[i, j, k, :] = able_area[random_choice]
                # randomly choose one point as negative sample
                random_choice = random.randint(0, disable_area.size()[0] - 1)
                prompt_list[i, j, multi_num, :] = disable_area[random_choice]
                label_list[i, j, multi_num] = torch.tensor([-1], dtype=torch.float)
    return prompt_list, label_list

def generate_box_resize_prompt(msk):
    # msk: (bd, t, 1, 1024, 1024)
    # prompt_list: (bd, t, 4)
    prompt_list = torch.zeros((msk.size()[0],msk.size()[1], 4),dtype=torch.float)
    for i in range(msk.size()[0]):
        for j in range(msk.size()[1]):
            # single_msk: (1024, 1024)
            single_msk = msk[i][j][0]
            able_area = torch.nonzero(single_msk)
            if able_area.size()[0] == 0:
                box_prompt = torch.tensor([-1,-1,-1,-1], dtype=torch.float)
                prompt_list[i,j,:] = box_prompt
            else:
                # print(now_msk[msk_type].squeeze(0).size())
                # print(able_area.size())
                random_choice = random.randint(0, able_area.size()[0] - 1)
                x_position = able_area[:,0]
                y_position = able_area[:,1]
                min_x = torch.min(x_position)
                max_x = torch.max(x_position)
                min_y = torch.min(y_position)
                max_y = torch.max(y_position)
                prompt_list[i,j,0] = min_x
                prompt_list[i,j,1] = min_y
                prompt_list[i,j,2] = max_x
                prompt_list[i,j,3] = max_y
                # left_up_point = able_area[0]
                # right_down_point = able_area[able_area.size()[0] - 1]
                # left_down_point = torch.tensor([right_down_point[0],left_up_point[1]], dtype=torch.float)
                # right_up_point = torch.tensor([left_up_point[0],right_down_point[1]], dtype=torch.float)
                # prompt_list[i,j,0] = left_up_point[0]
                # prompt_list[i,j,1] = left_up_point[1]
                # prompt_list[i,j,2] = right_down_point[0]
                # prompt_list[i,j,3] = right_down_point[1]

                # print(prompt_list[i,j])
    return prompt_list



# def generate_prompt(msk):
#     prompt_list = torch.zeros((msk.size()[0], 13, 2, msk.size()[5]), dtype=torch.float)
#     for b, single_msk in enumerate(msk):
#         # single_msk: (type, c, h, w, d)
#         for d in range(single_msk.size()[4]):
#             # now_msk: (type, c, h, w)
#             now_msk = single_msk[:,:,:,:,d]
#             able_area = torch.nonzero(now_msk).squeeze(1)
#             if able_area.size()[0] == 0:
#                 point_prompt = torch.tensor([-1, -1], dtype=torch.float)
#             else:
#                 random_choice = random.randint(0, able_area.size()[0] - 1)
#                 point_prompt = able_area[random_choice]
#             prompt_list[b, :, :, d] = point_prompt

#     return prompt_list
# msk = torch.zeros([1,1,5,5,1], dtype=torch.float)
# msk[0,0,2,2,0] = 1
# msk[0,0,2,4,0] = 2
# processed_msk = msk_preprocess(msk)
# processed_msk = torch.zeros([1,13 ,1,5,5,1], dtype=torch.float)
# generate_click_prompt(processed_msk)
