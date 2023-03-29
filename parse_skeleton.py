import xml.etree.ElementTree as ET

def parse_skeleton_xml(xml_path):
    root_node = ET.parse(xml_path).getroot()

    if not root_node.tag == 'annotation':
        raise Exception("No tag annotation found! Are you sure you are uploading correct xml file?")

    xml_parsed = {}
    xml_parsed['skeletons'] = []
    for elem in list(root_node):
        if elem.tag == 'folder':
            xml_parsed['folder'] = elem.text
        elif elem.tag == 'filename':
            xml_parsed['filename'] = elem.text
        elif elem.tag == 'size':
            xml_parsed['size'] = {'width': int(elem.findall('width')[0].text),
                                  'height': int(elem.findall('height')[0].text),
                                  'depth': int(elem.findall('depth')[0].text)}
        elif elem.tag == 'object':
            keypoints = []
            name = elem.findall('name')[0].text

            if name != 'skeleton1':
                raise Exception("Name of object is not skeleton!")
            keypoints_xml = elem.findall('keypoints')[0]
            bbox_xml = elem.findall('bndbox')[0]
            bounding_box = {'xmin': float(bbox_xml.findall('xmin')[0].text), 'ymin': float(bbox_xml.findall('ymin')[0].text),
                            'xmax': float(bbox_xml.findall('xmax')[0].text), 'ymax': float(bbox_xml.findall('ymax')[0].text)}
            for i in range(24):
                try:
                    to_append = {'x': float(keypoints_xml.findall('x{}'.format(i + 1))[0].text),
                                 'y': float(keypoints_xml.findall('y{}'.format(i + 1))[0].text),
                                 'v': int(keypoints_xml.findall('v{}'.format(i + 1))[0].text),}
                    keypoints.append(to_append)
                except:
                    raise Exception("Looks like your skeleton is broken - couldn't find {}th skeleton point".format(i + 1))

            xml_parsed['skeletons'].append({'keypoints' : keypoints, 'name': name, 'bounding_box': bounding_box})

    return xml_parsed