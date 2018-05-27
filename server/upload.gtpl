<html>
		<head>
				<title>Upload file</title>
		</head>
		<body>
				<form enctype="multipart/form-data" action="http://127.0.0.1:8080/upload" method="post">
						MRI DICOM: <input type="file" name="uploadfile" /><br>
						<input type="hidden" name="token" value="{{.}}"/>
						Segmenter Run Type: <input type="text" name="type"/><br>
						<input type="submit" value="upload" />
				</form>
		</body>
</html>
