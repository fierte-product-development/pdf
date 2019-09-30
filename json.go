// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// PDFファイルのパスの配列を受け取ってContentオブジェクトの配列をJsonで返す
// logをtrueにするとjsonファイルとついでにラインの解析結果をプロットしたpngファイルを作成する

package pdf

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"sync"
	"time"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

func parsePage(fileName string, fileIdx int, log bool) []Content {
	r, _ := Open(fileName)
	np := r.NumPage()
	doc := make([]Content, np)
	var wg sync.WaitGroup
	for i := 1; i <= np; i++ {
		wg.Add(1)
		go func(i int) {
			pg := r.Page(i)
			cont := pg.Contents()
			if log {
				mb := pg.V.Key("MediaBox")
				saveLinePng(&cont, &mb, fileIdx, i)
			}
			doc[i-1] = cont
			wg.Done()
		}(i)
	}
	wg.Wait()
	return doc
}

func saveLinePng(cont *Content, mbox *Value, fileIdx int, pageIdx int) {
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
	pngName := fmt.Sprintf("file%v_page%v.png", fileIdx+1, pageIdx)
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

// JSON receives an array of paths and outputs content object array and time as stdout.
func JSON(fileNames []string, log bool) {
	sTime := time.Now()
	docs := make([][]Content, len(fileNames))
	var wg sync.WaitGroup
	for i, fn := range fileNames {
		wg.Add(1)
		go func(fn string, i int) {
			docs[i] = parsePage(fn, i, log)
			wg.Done()
		}(fn, i)
	}
	wg.Wait()
	js, _ := json.MarshalIndent(docs, "", "  ")
	if log {
		ioutil.WriteFile("docs.json", js, os.ModePerm)
		eTime := time.Now()
		time := eTime.Sub(sTime).Seconds()
		fmt.Printf("complete. time: %fs\n", time)
	} else {
		os.Stdout.Write(js)
	}
}
