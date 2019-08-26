# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
#import sys
#sys.path.insert(0, 'path/to/Video-Bounding-Box-Labelling-Tool-master')
from tools.test_multi import *

import sys, termios, tty, os
import numpy as np
np.seterr(divide='ignore',invalid='ignore')



parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='../../data/tennis', help='datasets')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()



if __name__ == '__main__':

    # define the key
    def getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
    
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
    button_delay = 0.2

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup Model
    cfg = load_config(args)
    from custom import Custom
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)
    testmodel=[]
    testmodel.append(siammask)

    # Parse Image file
    img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))
    ims = [cv2.imread(imf) for imf in img_files]

    # Select ROI
    all_init_rect=[]

    while True:       
        print('Add new track, press "a"')
        print('Done, press "q"')
        char = getch()
        cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
        if (char == "a"):
            print('a')
            init_rect = cv2.selectROI('SiamMask', ims[0], False, False)
            x, y, w, h = init_rect
            if all_init_rect==[]:
                all_init_rect = [init_rect]
                label = np.array([1, x, y, w, h])
            else:
                all_init_rect = np.concatenate((all_init_rect,[init_rect]),axis=0)     
                l = np.array([1, x, y, w, h])
                label = np.append(label, l)        
        elif (char == "q"):   
            print('q')
            break        

    box_num = np.asarray(all_init_rect).shape[0]

    # init colors
    colorid = []
    for i in range(box_num):
        color = np.random.randint(0, high = 256, size = (3,))
        if colorid == []:
            colorid = [color]
        else:
            colorid = np.concatenate((colorid,[color]), axis=0) 

    toc = 0
    for f, im in enumerate(ims):
        try:
            tic = cv2.getTickCount()
            if f == 0:  # init               
                # all_state (input init bounding box, output state)
                all_state = []
                for i in range(box_num):
                    target_pos = np.array([all_init_rect[i][0] + all_init_rect[i][2] / 2, all_init_rect[i][1] + all_init_rect[i][3] / 2])
                    target_sz = np.array([all_init_rect[i][2], all_init_rect[i][3]])   
                    if all_state==[]:
                        state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
                        all_state = [state]
                    else:
                        state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
                        all_state = np.concatenate((all_state, [state]), axis=0)
    
            elif f > 0:  # tracking
                # all_pscore (input to the siamese_track_id to get pscore(in the state) )
                all_pscore = []
                for i in range(box_num):
                    state = siamese_track_id(all_state[i], im, mask_enable=True, refine_enable=True, device=device)  # track
                    pscore = state['pscore']
                    sort_pscore = np.sort(pscore)
                    if all_pscore == []:
                        all_pscore = [pscore]
                    else:
                        all_pscore = np.concatenate((all_pscore, [pscore]), axis=0)
                    all_state[i] = state

                # iou
                def iou(mask1, mask2):
                    intersection = np.logical_and(mask1, mask2)
                    union = np.logical_or(mask1, mask2)
                    iou = np.sum(intersection) / np.sum(union)
                    return iou

                # compare each max_pascore 
                search_id = []            
                unsearch_id = []
                for i in range(box_num):
                    if unsearch_id == []:
                        unsearch_id = [i]
                    else:
                        unsearch_id = np.append(unsearch_id, [i])

                while unsearch_id !=[] :
                    # find max pscore in each bounding box and put to max_pscore
                    max_pscore = []
                    for i in range(box_num):
                        if max_pscore == []:
                            max_pscore = np.amax(all_pscore[i])
                        else:
                            max_pscore = np.append(max_pscore, np.amax(all_pscore[i]))
                    # if pscore small then 0.8, delete tracker
                    if float(np.amax(max_pscore))<0.8 :                   
                        max_id = np.argmax(max_pscore)
                        for j in range(max_id,box_num-1):
                            all_state[j] =  all_state[j+1]
                            all_pscore[j] =  all_pscore[j+1]
                            max_pscore[j] = max_pscore[j+1]
                            colorid[j] = colorid[j+1]
                            all_init_rect[j] = all_init_rect[j+1]
                        box_num = box_num-1 
                        all_state = all_state[:-1]
                        all_pscore = all_pscore[:-1]
                        max_pscore = max_pscore[:-1]
                        colorid = colorid[:-1]
                        all_init_rect = all_init_rect[:-1]                       
                        unsearch_id = [x for x in unsearch_id if x != max_id]   
                        search_id = [x for x in search_id if x != max_id]
                        print('pscore<0.8 -> delete')
                        break
                    
                    # if two bounding boxes overlap, the lower pscore one get the second pscore
                    max_id = np.argmax(max_pscore)   
                    best_pscore_id = np.argmax(all_pscore[max_id]) 
                    state = siamese_get_mask(all_state[max_id], im, best_pscore_id, mask_enable=True, refine_enable=True, device=device)
                    mask = state['mask'] > state['p'].seg_thr
                    score = state['score']
                    if search_id !='NULL':           
                        for i in search_id:     
                            mask_s = all_state[int(i)]['mask'] > all_state[int(i)]['p'].seg_thr
                            while iou(mask, mask_s) > 0.85:          
                                all_pscore[max_id][best_pscore_id] = 0
                                best_pscore_id = np.argmax(all_pscore[max_id])
                                best_pscore_id = np.argmax(all_pscore[max_id])
                                if float(np.amax(all_pscore[max_id]))<0.8 : 
                                    print('iou>0.85 & second pscore<0.8 -> delete')
                                    for j in range(max_id,box_num-2):
                                        all_state[j] =  all_state[j+1]
                                        all_pscore[j] =  all_pscore[j+1]
                                        max_pscore[j] = max_pscore[j+1]
                                        colorid[j] = colorid[j+1]
                                        all_init_rect[j] = all_init_rect[j+1]
                                    box_num = box_num-1 
                                    all_state = all_state[:-1]
                                    all_pscore = all_pscore[:-1]
                                    max_pscore = max_pscore[:-1]
                                    colorid = colorid[:-1]
                                    all_init_rect = all_init_rect[:-1]  
                                    unsearch_id = [x for x in unsearch_id if x != max_id]   
                                    search_id = [x for x in search_id if x != max_id]
                                    break
                                else:
                                    print('iou>0.85 -> get next pscore')
                                    state = siamese_get_mask(all_state[max_id], im, best_pscore_id, mask_enable=True, refine_enable=True, device=device)
                                    mask = state['mask'] > state['p'].seg_thr
                                    continue

                    if state != []:
                        all_state[max_id] = state
                        all_pscore[max_id] = 0                    
                        search_id = np.append(search_id, [max_id])                   
                        unsearch_id = [x for x in unsearch_id if x != max_id]                

                # all_state
                all_location = []
                all_mask = []     
                all_score = []               
                for i in range(box_num):
                    location = all_state[i]['ploygon'].flatten()
                    location = np.asarray(location)
                    mask = all_state[i]['mask'] > all_state[i]['p'].seg_thr
                    score = all_state[i]['score']                           
                    if all_location == []:
                        all_location = [location]
                    else:
                        all_location = np.concatenate((all_location,[location]),axis=0)
                    if all_mask == []:
                        all_mask = [mask]
                    else:
                        all_mask = np.concatenate((all_mask,[mask]),axis=0)

                    if all_score == []:
                        all_score = [score]
                    else:
                        all_score = np.concatenate((all_score,[score]),axis=0)
                            
                # draw mask and bounding box
                for i in range(box_num):
                    im[:, :, 2] = (all_mask[i] > 0) * 255 + (all_mask[i] == 0) * im[:, :, 2]
                    cv2.rectangle(im, (int(all_location[i][0]), int(all_location[i][1])), (int(all_location[i][2]), int(all_location[i][3])), (int(colorid[i][0]), int(colorid[i][1]),int(colorid[i][2])), 3)
                    cv2.putText(im, str(all_score[i]), (int(all_location[i][0]), int(all_location[i][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (int(colorid[i][0]), int(colorid[i][1]),int(colorid[i][2])), 2)
                    # save label data to 'label'  [frame, x, y, w, h]                     
                    l = np.array([f+1, int(all_location[i][0]), int(all_location[i][1]), int(all_location[i][2])-int(all_location[i][0]), int(all_location[i][3])-int(all_location[i][1])])
                    label = np.append(label,l)

                cv2.imshow('SiamMask', im)
                #cv2.imwrite(str(f)+'.jpg',im)
                key = cv2.waitKey(1)
                if key > 0:
                    break

            toc += cv2.getTickCount() - tic

        except KeyboardInterrupt:
            print('\n')
            print('Continue, press "c"')
            print('Break, press "b"')
            char = getch()
            if (char == "c"):
                print('c')
                while True:       
                    print('Add new track, press "a"')
                    print('Done, press "q"')
                    char = getch()
                    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
                    if (char == "a"):
                        print('a')
                        init_rect = cv2.selectROI('SiamMask', ims[f], False, False)
                        x, y, w, h = init_rect
                        all_init_rect = np.concatenate((all_init_rect,[init_rect]),axis=0)     
                        l = np.array([f+1, x, y, w, h])
                        label = np.append(label, l)                     
                    elif (char == "q"):   
                        print('q')
                        break 
                
                add_box_num = np.asarray(all_init_rect).shape[0] - box_num

                for i in range(add_box_num):
                    color = np.random.randint(0, high = 256, size = (3,))
                    if colorid == []:
                        colorid = [color]
                    else:
                        colorid = np.concatenate((colorid,[color]), axis=0)

                for i in range(add_box_num):
                    target_pos = np.array([all_init_rect[i+box_num][0] + all_init_rect[i+box_num][2] / 2, all_init_rect[i+box_num][1] + all_init_rect[i+box_num][3] / 2])
                    target_sz = np.array([all_init_rect[i+box_num][2], all_init_rect[i+box_num][3]])               
                    if all_state==[]:
                        state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
                        all_state = [state]
                    else:
                        state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
                        all_state = np.concatenate((all_state, [state]), axis=0)

                box_num = np.asarray(all_init_rect).shape[0]
                continue
            elif (char == "b"):
                print('b')
                break

    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))


    label = label.reshape(int(label.shape[0]/5),5)
    
    for i in range(label.shape[0]):
        if i==0:
            fr = np.array(int(label[i][0]))
        else:
            f = np.array(int(label[i][0]))
            fr = np.append(fr,f)
    fr = np.asarray(fr)
    frame = np.max(fr)
    

    def createFolder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print ('Error: Creating directory. ' +  directory)


    print('\n')
    print('Create VOC file, press "1"')
    print('Create YOLO file, press "2"')
    print('Exit, press "3"')
    char = getch()



    if (char == "1"):
        print('Start to create VOC file...')
        from xml.dom import minidom
        import cv2    

        fo = args.base_path[5:]
        jpg_path = '/Video-Bounding-Box-Labelling-Tool-master' + fo + '/'
        xml_path = '/Video-Bounding-Box-Labelling-Tool-master/data/' + fo + '_xml/'

        createFolder(xml_path)
        
        for i in range(1,frame+1):
            jpgname = str(i).zfill(6)+'.jpg'
            img = cv2.imread(jpg_path+jpgname)
            w = im.shape[1]
            h = im.shape[0]
            d = im.shape[2]

            doc = minidom.Document()

            annotation = doc.createElement('annotation')
            doc.appendChild(annotation)

            folder = doc.createElement('folder')
            folder.appendChild(doc.createTextNode(fo))
            annotation.appendChild(folder)

            filename = doc.createElement('filename')
            filename.appendChild(doc.createTextNode(jpgname))
            annotation.appendChild(filename)
            
            filename = doc.createElement('path')
            filename.appendChild(doc.createTextNode(jpg_path + jpgname))
            annotation.appendChild(filename)
            
            source = doc.createElement('source')
            database = doc.createElement('database')
            database.appendChild(doc.createTextNode("Unknown"))
            source.appendChild(database)
            annotation.appendChild(source)

            size = doc.createElement('size')
            width = doc.createElement('width')
            width.appendChild(doc.createTextNode("%d" % w))
            size.appendChild(width)
            height = doc.createElement('height')
            height.appendChild(doc.createTextNode("%d" % h))
            size.appendChild(height)
            depth = doc.createElement('depth')
            depth.appendChild(doc.createTextNode("%d" % d))
            size.appendChild(depth)
            annotation.appendChild(size)
    
    
            segmented = doc.createElement('segmented')
            segmented.appendChild(doc.createTextNode("0"))
            annotation.appendChild(segmented)
            
            boxes = 0
            for j in range(label.shape[0]):
                if int(label[j][0])==i:
                    object = doc.createElement('object')
                    nm = doc.createElement('name')
                    nm.appendChild(doc.createTextNode('0'))
                    object.appendChild(nm)
                    pose = doc.createElement('pose')
                    pose.appendChild(doc.createTextNode("Unspecified"))
                    object.appendChild(pose)
                    truncated = doc.createElement('truncated')
                    truncated.appendChild(doc.createTextNode("0"))
                    object.appendChild(truncated)
                    difficult = doc.createElement('difficult')
                    difficult.appendChild(doc.createTextNode("0"))
                    object.appendChild(difficult)
                    bndbox = doc.createElement('bndbox')
                    xmin = doc.createElement('xmin')
                    xmin.appendChild(doc.createTextNode(str(label[j][1])))
                    bndbox.appendChild(xmin)
                    ymin = doc.createElement('ymin')
                    ymin.appendChild(doc.createTextNode(str(label[j][2])))
                    bndbox.appendChild(ymin)
                    xmax = doc.createElement('xmax')
                    xmax.appendChild(doc.createTextNode(str(int(label[j][1])+int(label[j][3]))))
                    bndbox.appendChild(xmax)
                    ymax = doc.createElement('ymax')
                    ymax.appendChild(doc.createTextNode(str(int(label[j][2])+int(label[j][4]))))
                    bndbox.appendChild(ymax)
                    object.appendChild(bndbox)
                    annotation.appendChild(object)
                    savefile = open(xml_path+str(i).zfill(6)+'.xml', 'w')
                    savefile.write(doc.toprettyxml())
                    savefile.close()    
        print('done!') 

    elif (char == '2'):
        print('Start to create YOLO file...')
        fi = open('/Video-Bounding-Box-Labelling-Tool-master/data/' + fo + '_yolo.txt','a')
        for i in range(1,frame+1):
            fi.write('Video-Bounding-Box-Labelling-Tool-master/data/'+str(args.base_path[5:])+'/'+str(i).zfill(6)+'.jpg')
            for j in range(label.shape[0]):
                if int(label[j][0])==i:
                    fi.write(' '+str(label[j][1])+','+str(label[j][2])+','+str(label[j][3])+','+str(label[j][4])+',1')
            fi.write('\n')
        print('done!')

        
    elif (char ==  '3'):
        exit