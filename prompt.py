import torch
import random

def msk_preprocess(msk):
    # msk: (b, c, h, w, d)
    # process_msk: (b, type, c, h, w, d)

    process_msk = torch.zeros((msk.size()[0], 13, msk.size()[1], msk.size()[2], msk.size()[3], msk.size()[4]),dtype=torch.float)
    for b in range(msk.size()[0]):
        single_msk = msk[b]
        for d in range(msk.size()[4]):
            now_msk = single_msk[:, :, :, d]
            for type in range(13):
                uni_msk = torch.zeros((msk.size()[2], msk.size()[3]))
                for h in range(msk.size()[2]):
                    for w in range(msk.size()[3]):
                        if now_msk[0][h][w] == type + 1:
                            uni_msk[h][w] = type + 1
                process_msk[b, type, 0, :, :, d] = uni_msk
                print(uni_msk)

    return process_msk



def generate_click_prompt(msk):
    # img: (b, c, h, w, d)
    # msk.shape: (b, type, c, h, w, d)
    # prompt_list: (b, type, 2, d)
    prompt_list = torch.zeros((msk.size()[0], 13, 2, msk.size()[5]),dtype=torch.float)
    for b, single_msk in enumerate(msk):
        # single_msk: (type, c, h, w, d)
        for d in range(single_msk.size()[4]):
            # now_msk: (type, c, h, w)
            now_msk = single_msk[:,:,:,:,d]
            for msk_type in range(13):
                # now_msk[msk_type]: (c, h, w)
                # able_area: (h*w, 2)
                able_area = torch.nonzero(now_msk[msk_type].squeeze(0))
                if able_area.size()[0] == 0:
                    point_prompt = torch.tensor([-1, -1], dtype=torch.float)
                else:
                   # print(now_msk[msk_type].squeeze(0).size())
                   # print(able_area.size())
                    random_choice = random.randint(0, able_area.size()[0] - 1)
                    point_prompt = able_area[random_choice]
                   # print(point_prompt)
                prompt_list[b, msk_type, :, d] = point_prompt

    # print(prompt_list)
    # print(prompt_list.shape)
    return prompt_list

# msk = torch.zeros([1,1,5,5,1], dtype=torch.float)
# msk[0,0,2,2,0] = 1
# msk[0,0,2,4,0] = 2
# processed_msk = msk_preprocess(msk)
# processed_msk = torch.zeros([1,13 ,1,5,5,1], dtype=torch.float)
# generate_click_prompt(processed_msk)
