��keras��tensorflow��caffe��ѧ��һ�飬ÿ�ֿ�ܶ��и����ʺϵ�Ӧ�ó���������keras�����ѧ�����Ľ��������ʹ��kerasѵ�����ѧϰģ�͡�

# һ����ȡ����
## ��1������׼��
����ʹ����д������ʶ�����ݿ�mnist��kerasͨ��mnist.load_data()���ɶ�ȡѵ�������Ͳ�������������ȡ���ݵ�ϸ�������ˡ�Ϊ��˵��keras��ζ�ȡ�Լ��ɼ������ݣ���Ϊmnist��10��label������Ӧ���Ƶ����ļ���[0 1 2 3 4 5 6 7 8 9]������mnist��ͼ���һ��Ϊ32x32���浽��Ӧ�ļ����С��ļ���Ŀ¼�ṹΪ��  
mnist.32x32/train/[0 1 2 3 4 5 6 7 8 9]  
mnist.32x32/test/[0 1 2 3 4 5 6 7 8 9]			  
## ��2�����ݶ�ȡ��������ǿ
keras�ṩ��ImageDataGenerator��ʵ�����ݶ�ȡ��������ǿ��ʹ��generator�ķ�����ȡ���ݵĺô��ǽ�ʡ�������Դ����Ϊ�������ܴ�ʱ�����޷�һ���Խ�����ȫ�����롣��Ҫ��������ǿ������ƽ�ơ���ת����ת������ü����˹�������  
���ݶ�ȡ�������£�  
    
    from keras.preprocessing.image import ImageDataGenerator  
    def load_mnist_data_from_directory_generator(dirpath, target_size = (256, 256), batch_size=32, color_mode='rgb', class_mode='categorical', classes=None):  
        train_datagen = ImageDataGenerator(rescale=1 / 255.0)  
        test_datagen = ImageDataGenerator(rescale=1 / 255.0)
        train_generator = train_datagen.flow_from_directory(
            '{0}/train'.format(dirpath),
            target_size=target_size,
            batch_size=batch_size,
            class_mode=class_mode,
            color_mode=color_mode,
            classes=classes)

        test_generator = test_datagen.flow_from_directory(
            '{0}/test'.format(dirpath),
            target_size=target_size,
            batch_size=batch_size,
            class_mode=class_mode,
            color_mode=color_mode,
            classes=classes)
        return train_generator, test_generator

	
# ����ģ�����  
������LeNetΪ�������ú���ʽAPI���ģ�͡�
	
	def LeNet(input_shape, num_classes):
        inputs = Input(input_shape, name='input_1')
        x = Conv2D(6, (5, 5), activation='relu', name='conv1')(inputs)
        x = MaxPool2D((2, 2), name='max_pooling2d_1')(x)
        x = Conv2D(16, (5, 5), activation='relu', name='conv2')(x)
        x = MaxPool2D((2, 2), name='max_pooling2d_2')(x)
        x = Flatten(name='flatten_1')(x)
        x = Dense(120, activation='relu', name='fc1')(x)
        x = Dense(84, activation='relu', name='fc2')(x)
        predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
        model =  Model(inputs=inputs, outputs=predictions)
        return model
	
# ����ģ�ͱ���  
��Ҫָ���Ż�������ʧ����������������

	model = LeNet(input_shape, num_classes)
    optimizer = optimizers.SGD(lr=0.01)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
	
# �ġ�ģ��ѵ��
�˴���Ҫָ��epochs��batch_size��flow_from_directory��ָ����ͨ��TensorBoardʵ�ֿ��ӻ���ͨ��ModelCheckpointʵ�ּ���ı��棬���ڻָ��жϵ�ģ��ѵ����
	
	tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir)
    checkpointer = keras.callbacks.ModelCheckpoint(filepath=ckpt_filepath)

    history = model.fit_generator(generator=train_generator,
                        steps_per_epoch=len(train_generator),
                        epochs=epochs,
                        verbose=1,
                        callbacks=[tensorboard, checkpointer],
                        validation_data=test_generator,
                        validation_steps=len(test_generator),
                        initial_epoch=initial_epoch)
	
tensorboard���ӻ������  
*graphs*:
![Image text](resources/tf_graphs.png)
  
*scalars*:  
![Image text](resources/tf_acc.jpg)
![Image text](resources/tf_loss.jpg)  
![Image text](resources/tf_val_acc.jpg)  
![Image text](resources/tf_val_loss.jpg)  

# �塢ģ������
����ģ�͵�����

	score_evaluate = model.evaluate_generator(generator=test_generator, steps=len(test_generator))
    print(score_evaluate)
						
# ����ģ�ͱ����뵼��
	model.save(model_name)
	model =  keras.models.load_model(model_name)
	
# �ߡ���������
## ��1����checkpoint�ָ��жϵ�ģ��ѵ��
����ģ�ͱ���Ϊckpt_filepath = ckpt_dir + '/weights.{epoch:02d}-{val_loss:.2f}.h5'���ָ��жϵ�ģ��ѵ����

	ckpt_filelist = glob.glob('{0}/weights.*.h5'.format(ckpt_dir))
    if len(ckpt_filelist) == 0:
        model = mnist_lenet(input_shape, num_classes)
    else:
        filename = ckpt_filelist[-1]
        model = keras.models.load_model(filepath=filename)
        initial_epoch = int(filename.split('weights.')[-1].split('-')[0])
        print('restore from {0}, initial_epoch = {1}'.format(filename, initial_epoch))
        
## ��2��ģ�Ϳ��ӻ�
������tensorboard��������ʹ��model.summary()��plot_model��  	
*plot_model���*��
![Image text](resources/plot_model.png)

# �ˡ�learning_keras˼ά��ͼ
![Image text](resources/learning_keras.png)	
	    
	