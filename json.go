// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// PDFファイルのパスの配列を受け取ってContentオブジェクトの配列をjsonで返す
// toStdoutがtrueの場合jsonを標準出力に流す
// falseの場合jsonファイルとついでにラインの解析結果をプロットしたpngファイルを出力する

package pdf

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sync"
	"time"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

func parsePage(filePath string, toStdout bool) []Content {
	r, _ := Open(filePath)
	np := r.NumPage()
	doc := make([]Content, np)
	var wg sync.WaitGroup
	for i := 1; i <= np; i++ {
		wg.Add(1)
		go func(i int) {
			pg := r.Page(i)
			cont := pg.Contents()
			if !toStdout {
				mb := pg.V.Key("MediaBox")
				saveLinePng(&cont, &mb, filePath, i)
			}
			doc[i-1] = cont
			wg.Done()
		}(i)
	}
	wg.Wait()
	return doc
}

func saveLinePng(cont *Content, mbox *Value, filePath string, pageIdx int) {
	plt, _ := plot.New()
	plt.Add(plotter.NewGrid())
	plt.X.Min = mbox.Index(0).Float64()
	plt.Y.Min = mbox.Index(1).Float64()
	plt.X.Max = mbox.Index(2).Float64()
	plt.Y.Max = mbox.Index(3).Float64()
	for _, t := range cont.Table {
		for _, c := range t.Cell {
			draw(plt, [4]float64{c.Min.X, c.Max.X, c.Max.Y, c.Max.Y}, 0, 0)
			draw(plt, [4]float64{c.Max.X, c.Max.X, c.Min.Y, c.Max.Y}, 0, 0)
			draw(plt, [4]float64{c.Min.X, c.Max.X, c.Min.Y, c.Min.Y}, 0, 0)
			draw(plt, [4]float64{c.Min.X, c.Min.X, c.Min.Y, c.Max.Y}, 0, 0)
		}
	}
	for _, l := range cont.Line {
		xy := l.ToXY()
		draw(plt, [4]float64{xy[0].X, xy[1].X, xy[0].Y, xy[1].Y}, 2, 2)
	}
	w := vg.Length(plt.X.Max/100) * vg.Inch
	h := vg.Length(plt.Y.Max/100) * vg.Inch
	pngName := fmt.Sprintf("%v_page%v.png", removeExtension(filePath), pageIdx)
	plt.Save(w, h, pngName)
}

func draw(plt *plot.Plot, points [4]float64, color int, dashes int) {
	min := plotter.XY{X: points[0], Y: points[2]}
	max := plotter.XY{X: points[1], Y: points[3]}
	l, s, _ := plotter.NewLinePoints(plotter.XYs{min, max})
	l.Color = plotutil.Color(color)
	l.Dashes = plotutil.Dashes(dashes)
	s.Color = plotutil.Color(color)
	plt.Add(l, s)
}

func removeExtension(filePath string) string {
	fileName := filepath.Base(filePath)
	extension := filepath.Ext(fileName)
	return fileName[:len(fileName)-len(extension)]
}

// JSON receives an array of paths and outputs array of content objects as JSON.
func JSON(filePaths []string, toStdout bool) {
	sTime := time.Now()
	docs := make([][]Content, len(filePaths))
	var wg sync.WaitGroup
	for i, fn := range filePaths {
		wg.Add(1)
		go func(fn string, i int) {
			docs[i] = parsePage(fn, toStdout)
			wg.Done()
		}(fn, i)
	}
	wg.Wait()
	js, _ := json.MarshalIndent(docs, "", "  ")
	if toStdout {
		os.Stdout.Write(js)
	} else {
		ioutil.WriteFile("docs.json", js, os.ModePerm)
		eTime := time.Now()
		time := eTime.Sub(sTime).Seconds()
		fmt.Printf("complete. time: %fs\n", time)
	}
}
