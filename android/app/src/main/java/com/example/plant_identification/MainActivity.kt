package com.example.plant_identification

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.*
import kotlinx.coroutines.launch
import java.io.ByteArrayOutputStream
import java.io.InputStream
import com.example.plant_identification.ui.theme.Plant_identificationTheme
import androidx.compose.runtime.remember
import coil.compose.rememberImagePainter
import coil.request.ImageRequest
import androidx.activity.result.contract.ActivityResultContracts
import android.widget.Toast
import kotlinx.coroutines.delay
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.PhotoCamera
import androidx.compose.material.icons.filled.Spa
import androidx.compose.material.icons.filled.Upload
import androidx.compose.foundation.background
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.CardDefaults
import androidx.compose.ui.layout.ContentScale



// 创建模拟API实现
class MockPlantIdentificationApi : PlantIdentificationApi {
    override suspend fun predict(imageData: ImageData): PredictionResult {
        // 模拟网络延迟
        delay(1000)
        // 返回硬编码的预测结果
        return PredictionResult(
            plant_name = "玫瑰",
            confidence = 0.95f
        )
    }
}


// 定义API接口
interface PlantIdentificationApi {
    @POST("/predict")
    suspend fun predict(@Body imageData: ImageData): PredictionResult
}

// 数据类定义
data class ImageData(val image: String) // Base64编码的图片数据
data class PredictionResult(val plant_name: String, val confidence: Float)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            Plant_identificationTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    PlantIdentificationScreen()
                }
            }
        }
    }
}

@Composable
fun PlantIdentificationScreen() {
    var imageUri by remember { mutableStateOf<Uri?>(null) }
    var predictionResult by remember { mutableStateOf<PredictionResult?>(null) }
    var isLoading by remember { mutableStateOf(false) }
    val coroutineScope = rememberCoroutineScope()
    val context = LocalContext.current

    // 使用模拟API

    val retrofit = Retrofit.Builder()
        .baseUrl("http://192.168.2.101:5000/") // 替换为你的Flask服务器地址
        .addConverterFactory(GsonConverterFactory.create())
        .build()

    val api = retrofit.create(PlantIdentificationApi::class.java)
    //val api = MockPlantIdentificationApi()
        // 图片选择器
        val imagePicker = rememberLauncherForActivityResult(
            contract = ActivityResultContracts.GetContent(),
            onResult = { uri ->
                uri?.let {
                    imageUri = it
                }
            }
        )

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp)
                .background(Color(0xFFF0F4E3)), // 添加浅绿色背景
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            // 显示选择的图片
            imageUri?.let { uri ->
                Card(
                    modifier = Modifier
                        .size(200.dp)
                        .padding(8.dp),
                    shape = RoundedCornerShape(16.dp), // 圆角卡片
                    elevation = CardDefaults.cardElevation(defaultElevation = 4.dp) // 使用CardElevation
                ) {
                    Image(
                        painter = rememberImagePainter(
                            data = uri,
                            builder = {
                                crossfade(true)
                            }
                        ),
                        contentDescription = "Selected Image",
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Crop // 图片裁剪填充
                    )
                }
            }

            // 显示预测结果
            predictionResult?.let { result ->
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(8.dp),
                    shape = RoundedCornerShape(16.dp), // 圆角卡片
                    elevation = CardDefaults.cardElevation(defaultElevation = 4.dp), // 使用CardElevation
                    colors = CardDefaults.cardColors(containerColor = Color(0xFFE8F5E9)) // 浅绿色背景
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp),
                        horizontalAlignment = Alignment.CenterHorizontally, // 水平居中
                        verticalArrangement = Arrangement.Center // 垂直居中
                    ) {
                        Icon(
                            imageVector = Icons.Default.Spa, // 植物图标
                            contentDescription = "Plant Icon",
                            tint = Color(0xFF4CAF50), // 绿色图标
                            modifier = Modifier.size(32.dp)
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = "植物名称: ${result.plant_name}",
                            color = Color(0xFF2E7D32), // 深绿色文字
                            fontSize = 18.sp,
                            fontWeight = FontWeight.Bold
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = "置信度: ${result.confidence}",
                            color = Color(0xFF43A047), // 绿色文字
                            fontSize = 16.sp
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // 选择图片按钮
            Button(
                onClick = {
                    imagePicker.launch("image/*")
                },
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF4CAF50)), // 绿色按钮
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 32.dp)
            ) {
                Icon(
                    imageVector = Icons.Default.PhotoCamera, // 相机图标
                    contentDescription = "Select Image",
                    modifier = Modifier.size(24.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text("选择图片", color = Color.White)
            }

            Spacer(modifier = Modifier.height(8.dp))

            // 上传按钮
            Button(
                onClick = {
                    imageUri?.let { uri ->
                        coroutineScope.launch {
                            isLoading = true
                            try {
                                val bitmap = uriToBitmap(context, uri)
                                bitmap?.let {
                                    val base64Image = bitmapToBase64(it)
                                    val imageData = ImageData(base64Image)
                                    predictionResult = api.predict(imageData)
                                }
                            } catch (e: Exception) {
                                Toast.makeText(context, "请求失败: ${e.message}", Toast.LENGTH_SHORT).show()
                            } finally {
                                isLoading = false
                            }
                        }
                    }
                },
                enabled = !isLoading && imageUri != null,
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF4CAF50)), // 浅绿色按钮
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 32.dp)
            ) {
                if (isLoading) {
                    CircularProgressIndicator(color = Color.White)
                } else {
                    Icon(
                        imageVector = Icons.Default.Upload, // 上传图标
                        contentDescription = "Upload Image",
                        modifier = Modifier.size(24.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("上传图片并识别", color = Color.White)
                }
            }
        }
    }
// 将Uri转换为Bitmap
private fun uriToBitmap(context: Context, uri: Uri): Bitmap? {
    return try {
        val inputStream: InputStream? = context.contentResolver.openInputStream(uri)
        BitmapFactory.decodeStream(inputStream)
    } catch (e: Exception) {
        e.printStackTrace()
        null
    }
}

// 将Bitmap转换为Base64字符串
private fun bitmapToBase64(bitmap: Bitmap, format: Bitmap.CompressFormat = Bitmap.CompressFormat.PNG): String {
    val byteArrayOutputStream = ByteArrayOutputStream()
    bitmap.compress(format, 100, byteArrayOutputStream)
    val byteArray = byteArrayOutputStream.toByteArray()
    return android.util.Base64.encodeToString(byteArray, android.util.Base64.DEFAULT)
}