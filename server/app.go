package main

import (
	"bytes"
	"crypto/md5"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"text/template"
	"time"
)

// Defaults
const (
	PORT   = ":8080"
	MAXMEM = 32 << 20
)

// Struct to marshal uploads and results
type Submission struct {
	Upload  []byte `json:"upload"`
	RunType string `json:"type"`
	Output  []byte `json:"output"`
}

// Function to kick off build pipeline and retrieve results
func (s *Submission) Exec() {
	// TODO:
	// Store upload into a temporary file
	// Execute bash script to run ML on upload
	// Get result and store into output
}

// Shut server down on key errors
func checkErr(err error) {
	if err != nil {
		panic(err)
	}
}

// Convert uploaded file into byte slice
func convertToBytes(r io.Reader) ([]byte, error) {
	buf := bytes.NewBuffer(nil)
	if _, err := io.Copy(buf, r); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// Handler for uploading files
func uploadHandler(w http.ResponseWriter, r *http.Request) {
	// Display the Go template file for client uploading
	if r.Method == "GET" {
		crutime := time.Now().Unix()
		h := md5.New()
		io.WriteString(h, strconv.FormatInt(crutime, 10))
		token := fmt.Sprintf("%x", h.Sum(nil))

		t, _ := template.ParseFiles("upload.gtpl")
		t.Execute(w, token)
	} else {
		// Parse form and push dicom into build pipeline
		err := r.ParseMultipartForm(MAXMEM)
		var JSON []byte
		checkErr(err)
		f, h, err := r.FormFile("upload")
		checkErr(err)
		defer f.Close()
		ext := strings.Split(h.Filename, ".")[1]
		if ext != "dcm" {
			JSON, _ = json.Marshal(map[string]interface{}{
				"status":  500,
				"message": "Need Dicom file",
			})
		} else {
			s := Submission{}
			s.Upload, err = convertToBytes(f)
			checkErr(err)
			s.RunType = r.FormValue("type")
			s.Exec()
			JSON, _ = json.Marshal(map[string]interface{}{
				"status":     200,
				"message":    "Successfully ran segmenter",
				"submission": s,
			})
		}
		w.Write(JSON)
	}
}

func main() {
	// Basic server settings
	http.HandleFunc("/upload", uploadHandler)
	http.ListenAndServe(PORT, nil)
}
