# JIGSAWS 官方下载接入说明（原始 kinematics）

## 1. 为什么需要这一步

JIGSAWS 原始数据（`kinematics + transcriptions`）由官方表单发放链接。  
公开镜像常见情况是只有视频，因此程序会进入 `video_fallback` 模式。

## 2. 官方入口

- 表单地址：`https://www.cs.jhu.edu/~los/jigsaws/info.php`
- 需要填写姓名、邮箱、单位并通过 reCAPTCHA
- 官方会把下载链接发送到你的邮箱

## 3. 在本项目中一键导入

收到邮件链接后执行：

```bash
python3 -m imu_intent.fetch_jigsaws_official \
  --suturing-url "<官方邮件链接1>" \
  --knot-url "<官方邮件链接2>" \
  --needle-url "<官方邮件链接3>"
```

也支持本地 zip：

```bash
python3 -m imu_intent.fetch_jigsaws_official \
  --suturing-zip "/abs/path/Suturing.zip" \
  --knot-zip "/abs/path/Knot_Tying.zip" \
  --needle-zip "/abs/path/Needle_Passing.zip"
```

## 4. 验证是否已经替换 fallback

```bash
python3 -m imu_intent.verify_jigsaws_layout \
  --config config/imu_multidataset.toml \
  --output logs/jigsaws_layout_check.json
```

如果输出 `mode = "kinematics"`，表示已成功切换到原始源。  
随后重新训练模型即可。

