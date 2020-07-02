import os
import argparse
import tensorflow as tf
try:
    from tqdm import tqdm
    use_tqdm = True
except ImportError:
    print('pip install tqdm - to use progress bar.')
    use_tqdm = False
from multiprocessing.dummy import Pool


def edit_lines(file_name, line_no, new_line, new_file_name):
    if not isinstance(line_no, list):
        line_no = [line_no]
        new_line = [new_line]
    f = open(file_name, 'r')
    n = open(new_file_name, 'w')
    for i, line in enumerate(f):
        line = line.strip()
        if (i+1) in line_no:
            print(new_line[line_no.index(i+1)], file=n)
        else:
            print(line, file=n)
    f.close()
    n.close()


def parse_func(serialized_example):
    features = tf.io.parse_single_example(serialized_example,
                                       features={
                        'image/height': tf.io.FixedLenFeature([], tf.int64),
                        'image/width': tf.io.FixedLenFeature([], tf.int64),
                        'image/encoded': tf.io.FixedLenFeature([], tf.string),
                        'image/format': tf.io.FixedLenFeature([], tf.string),
                        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
                        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
                        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
                        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
                        'image/object/class/text': tf.io.VarLenFeature(tf.string),
                        'image/object/class/label': tf.io.VarLenFeature(tf.int64)
    })
    image = features['image/encoded']
    image_format = features['image/format']
    height = tf.cast(features['image/height'], tf.int32)
    width = tf.cast(features['image/height'], tf.int32)
    xmin = tf.sparse.to_dense(features['image/object/bbox/xmin'])
    xmax = tf.sparse.to_dense(features['image/object/bbox/xmax'])
    ymin = tf.sparse.to_dense(features['image/object/bbox/ymin'])
    ymax = tf.sparse.to_dense(features['image/object/bbox/ymax'])
    label = tf.sparse.to_dense(features['image/object/class/label'])
    return image, image_format, height, width, xmin, xmax, ymin, ymax, label


def read_record(record_file):
    dataset = tf.data.TFRecordDataset([record_file])
    dataset = dataset.map(parse_func)
    iterator = dataset.__iter__()
    next_element = next(iterator)
    try:
        while(True):
            (image, image_format, height, width,
             xmin, xmax, ymin, ymax, label) = next_element
            yield (image.numpy(), image_format.numpy().decode('ascii'), height.numpy(),
                    width.numpy(), xmin.numpy(), xmax.numpy(), ymin.numpy(), ymax.numpy(),
                    label.numpy())
    except tf.errors.OutOfRangeError:
        pass


def make_train_val(train_records, val_records, images_dir,
                   train_dir, val_dir, prefixes=None, threads=10,
                   batch_size=100):
    train_file = open(train_dir, 'w')
    val_file = open(val_dir, 'w')
    prefixes = prefixes or \
        ['train_{}_'.format(i) for i in range(len(train_records))] + \
        ['val_{}_'.format(i) for i in range(len(val_records))]
    for record, prefix in zip(train_records + val_records, prefixes):
        convert_record(record, images_dir, prefix,
                       train_dir, val_dir, threads,
                       batch_size, train_file, val_file)
    train_file.close()
    val_file.close()
    

def convert_record(record_file, images_dir, prefix, train_dir,
                   val_dir, threads, batch_size, train_file, val_file):
    print('Processing : {}'.format(record_file))
    args_list = []
    pool = Pool(threads)
    iterator = enumerate(read_record(record_file))
    if use_tqdm:
        iterator = tqdm(iterator)
    for i, example in iterator:
        image_path = '{}{}{}'.format(images_dir, prefix, i)
        image_format = example[1]
        args_list.append((example,
                          image_path))
        if 'train' in image_path:
            print('data/{}.{}'.format(image_path, image_format),
                  file=train_file)
        elif 'val' in image_path:
            print('data/{}.{}'.format(image_path, image_format),
                  file=val_file)
        if len(args_list) == batch_size:
            _ = pool.map(write_example, args_list)
            args_list = []
    if len(args_list) > 0:
        _ = pool.map(write_example, args_list)
    pool.close()
    pool.join()


def write_example(args):
    example, image_path = args
    image, image_format, height, width, xmin, xmax, ymin, ymax, label = example
    with open(image_path + '.' + image_format, 'wb') as f:
        f.write(image)
    with open(image_path + '.' + 'txt', 'w') as f:
        for x1, x2, y1, y2, l in zip(xmin, xmax, ymin, ymax, label):
            print('{} {} {} {} {}'.format(l-1, (x1+x2)/2,
                                          (y1+y2)/2, (x2-x1),
                                          (y2-y1)), file=f)

def make_cfg(source_file, batch_size, num_classes,
             destination_file='yolo-obj.cfg'):
    line_no = [3, 4, 643, 729, 816, 636, 722, 809] 
    new_line = ['batch={}'.format(batch_size),
                'subdivisions={}'.format(batch_size),
                'classes={}'.format(num_classes),
                'classes={}'.format(num_classes),
                'classes={}'.format(num_classes),
                'filters={}'.format(3*num_classes + 15),
                'filters={}'.format(3*num_classes + 15),
                'filters={}'.format(3*num_classes + 15)]
    edit_lines(source_file, line_no,
               new_line, destination_file)


def make_obj(label_map, obj_names, obj_data,
             train_dir, val_dir, backup_dir):
    class_dict = make_obj_names(label_map, obj_names)
    make_obj_data(obj_data, obj_names, len(class_dict),
                  train_dir, val_dir, backup_dir)
    return class_dict


def make_obj_names(label_map, obj_names='obj.names'):
    f = open(label_map, 'r')
    n = open(obj_names, 'w')
    class_dict = {}
    for line in f:
        if 'name:' not in line:
            continue
        name = line.split('name:')[-1].strip().replace("'", "")
        class_dict[name] = len(class_dict)
        print(name, file=n)
    f.close()
    n.close()
    return class_dict


def make_obj_data(obj_data, obj_names, num_classes,
                  train_dir, val_dir, backup_dir):
    f = open(obj_data, 'w')
    print('classes = {}'.format(num_classes), file=f)
    print('train = data/{}'.format(train_dir), file=f)
    print('valid = data/{}'.format(val_dir), file=f)
    print('names = data/{}'.format(obj_names), file=f)
    print('backup = {}'.format(backup_dir), file=f)
    f.close()


def convert(train_records, val_records,
            images_dir='obj/', train_dir='train.txt',
            val_dir='val.txt', source_file='yolov3-spp.cfg',
            batch_size=64, label_map='label_map.pbtxt',
            obj_names='obj.names', obj_data='obj.data',
            backup_dir='backup/', destination_file='yolo-obj.cfg',
            threads=10, thread_batch_size=100):
    print('Making obj files')
    class_dict = make_obj(label_map, obj_names, obj_data,
                          train_dir, val_dir, backup_dir)
    print('Making cfg file')
    make_cfg(source_file, batch_size, len(class_dict), destination_file)
    os.makedirs(images_dir, exist_ok=True)
    print('Reading tf record files')
    make_train_val(train_records, val_records, images_dir,
                   train_dir, val_dir, threads=threads,
                   batch_size=thread_batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_records',
                        help="Comma seperated list of training tf records,\
                              eg. train1.record, train2.record",
                        type=str, default='train.record')
    parser.add_argument('--val_records',
                        help="Comma seperated list of validation tf records,\
                              eg. val1.record, val2.record",
                        type=str, default='val.record') 
    parser.add_argument('--images_dir',
                        help='Folder to put images in',
                        type=str, default='obj/')
    parser.add_argument('--train_dir',
                        help='Path of txt file to put names of train images',
                        type=str, default='train.txt')
    parser.add_argument('--val_dir',
                        help='Path of txt file to put names of val images',
                        type=str, default='val.txt')
    parser.add_argument('--source_file',
                        help='Path of source yolo config file',
                        type=str, default='yolov3-spp.cfg')
    parser.add_argument('--label_map',
                        help='Path of tf object detection api label_map.pbtxt',
                        type=str, default='label_map.pbtxt')
    parser.add_argument('--obj_names',
                        help='Path of to make obj_names file for yolo',
                        type=str, default='obj.names')
    parser.add_argument('--obj_data',
                        help='Path of to make obj_data file for yolo',
                        type=str, default='obj.data')
    parser.add_argument('--backup_dir',
                        help='Path of backup_dir for yolo models by darknet',
                        type=str, default='backup/')
    parser.add_argument('--batch_size', help='Training batch size',
                        type=int, default=64)
    parser.add_argument('--thread_batch_size', help='Thread batch size',
                        type=int, default=100)
    parser.add_argument('--threads', help='Training batch size', type=int, default=10)
    parser.add_argument('--destination_file',
                        help='Path of config file for new yolo model',
                        type=str, default='yolo-obj.cfg')
    args = parser.parse_args()
    convert(args.train_records.replace(' ', '').split(','),
            args.val_records.replace(' ', '').split(','),
            images_dir=args.images_dir, train_dir=args.train_dir,
            val_dir=args.val_dir, source_file=args.source_file,
            batch_size=args.batch_size, label_map=args.label_map,
            obj_names=args.obj_names, obj_data=args.obj_data,
            backup_dir=args.backup_dir,
            destination_file=args.destination_file,
            threads=args.threads,
            thread_batch_size=args.thread_batch_size)
 
