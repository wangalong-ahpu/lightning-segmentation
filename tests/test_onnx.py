import os
import cv2
from tqdm import tqdm
import numpy as np


class SegmentationONNX:
    def __init__(self, model_path: str):
        """
        初始化分割模型推理器
        
        Args:
            model_path (str): ONNX模型路径
            input_size (tuple): 模型输入尺寸 (height, width)
        """
        self.model_type = 'onnx' if model_path.endswith('.onnx') else 'om'
        try:
            if self.model_type == 'onnx':
                import onnxruntime as ort
                self.session = ort.InferenceSession(model_path,
                                                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"],)
            else:
                raise TypeError("暂不支持该模型进行推理")
        except Exception as e:
            raise RuntimeError(f"无法加载模型: {e}")
        
        # 获取模型输入信息
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = self.session.get_inputs()[0].shape[2:]
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像以适配模型输入
        
        Args:
            image (numpy.ndarray): 输入图像
            
        Returns:
            numpy.ndarray: 预处理后的图像
        """
        # 保存原始尺寸
        self.original_shape = image.shape[:2]  # (height, width)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_size[1], self.input_size[0]))
        image = image.astype(np.float32) / 255.0
        # 转换为CHW格式并添加批次维度
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        image = np.expand_dims(image, axis=0)   # 添加批次维度
        
        return image
    
    def postprocess(self, output: np.ndarray) -> np.ndarray:
        """
        后处理模型输出
        
        Args:
            output (numpy.ndarray): 模型输出
            
        Returns:
            numpy.ndarray: 处理后的输出
        """
        # 如果输出是多通道的，获取类别预测
        if len(output.shape) > 2:
            # 对于分割任务，通常在通道维度上取argmax得到类别
            if output.shape[1] > 1:  # 多类别
                output = np.argmax(output, axis=1)
            else:  # 单通道输出
                output = (output > 0.5).astype(np.uint8)
        
        output = np.squeeze(output, axis=0)
        output = cv2.resize(
            output.astype(np.uint8), 
            (self.original_shape[1], self.original_shape[0]), 
            interpolation=cv2.INTER_NEAREST
        )   # 调整回原始尺寸
        
        return output
    
    def infer(self, image: np.ndarray) -> np.ndarray:
        """
        对单张图像进行推理
        
        Args:
            image (numpy.ndarray): 输入图像
            
        Returns:
            numpy.ndarray: 分割结果
        """
        input_image = self.preprocess(image)    # 预处理
        try:
            outputs = self.session.run(None, {self.input_name: input_image})
            prediction = outputs[0]
        except Exception as e:
            raise RuntimeError(f"模型推理时出错: {e}")
        
        result = self.postprocess(prediction)   # 后处理
        
        return result
    
    def visualize_result(self, original_image: np.ndarray, prediction: np.ndarray, 
                         colors: list = None) -> np.ndarray:
        """
        可视化预测结果
        
        Args:
            original_image (numpy.ndarray): 原始图像
            prediction (numpy.ndarray): 预测结果
            colors (list): 类别颜色列表
            
        Returns:
            numpy.ndarray: 可视化结果图像
        """
        # 转换颜色空间
        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        # 创建彩色mask
        if len(np.unique(prediction)) <= 2:  # 二值图像
            mask = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)  # 创建绿色mask
            mask[prediction > 0] = [0, 255, 0]
        else:  # 多类别分割
            colored_mask = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)  # 创建彩色map
            if colors is None:
                colors = [
                    [0, 0, 0],       # 背景 - 黑色
                    [255, 0, 0],     # 类别1 - 红色
                    [0, 255, 0],     # 类别2 - 绿色
                    [0, 0, 255],     # 类别3 - 蓝色
                    [255, 255, 0],   # 类别4 - 青色
                    [255, 0, 255],   # 类别5 - 紫色
                    [0, 255, 255],   # 类别6 - 黄色
                ]
            for i in range(min(len(colors), prediction.max() + 1)): # 为不同类别分配不同颜色
                colored_mask[prediction == i] = colors[i]
            mask = colored_mask

        overlay = cv2.addWeighted(original_image_rgb, 0.7, mask, 0.3, 0)    # 叠加mask到原始图像
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)  # 转换回BGR颜色空间
        
        return overlay_bgr


if __name__ == '__main__':
    Segmentation = SegmentationONNX(model_path="outputs/checkpoints/best-model-epoch=149-valid_dataset_iou=0.9731.onnx")    # 初始化模型
    
    input_folder = "datasets/tests"
    out_folder = "datasets/test_results"
    os.makedirs(out_folder, exist_ok=True)
    
    fn_list = [fn for fn in sorted(os.listdir(input_folder)) if fn.endswith((".jpg",".png",".jpeg",'.bmp'))]
    for fn in tqdm(fn_list):
        img_path = os.path.join(input_folder,fn)
        name_without_ext = os.path.splitext(fn)[0]
        original_image = cv2.imread(img_path)
        prediction = Segmentation.infer(original_image)
        result_image = Segmentation.visualize_result(original_image, prediction)
        cv2.imwrite(os.path.join(out_folder, f"result_{name_without_ext}.png"), result_image)
        cv2.imwrite(os.path.join(out_folder, f"mask_{name_without_ext}.png"), prediction)
    print(f"处理结果已保存到: {out_folder}")