qwe = tf.reduce_sum(image, axis=2)
  xsum = tf.reduce_sum(qwe, axis=0)
  itemidx = tf.where(xsum != 0)
  ysum = tf.reduce_sum(qwe, axis=1)
  itemidy = tf.where(ysum != 0)
  image = tf.image.crop_to_bounding_box(image, tf.reduce_min(itemidx), tf.reduce_min(itemidy), tf.reduce_max(itemidx) - tf.reduce_min(itemidx), tf.reduce_max(itemidy) - tf.reduce_min(itemidy))
  zx = image
  image = tf.image.resize_images(image, [224, 224])
  image /= 255
  image.set_shape([224, 224, 3])