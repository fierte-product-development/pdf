// PDFファイルのパスの配列を受け取ってContentオブジェクトの配列をjsonで返す
// toFileがTrueの場合jsonファイルとついでにラインの解析結果をプロットしたpngファイルを出力する

package pdf

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/rsc.io/pdf/core"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

type Page struct {
	Contents core.Content
	MediaBox core.BoundingBox
}

func NewPage(pg core.Page) *Page {
	p := new(Page)
	p.Contents = pg.Contents()
	p.MediaBox = pg.MediaBox()
	return p
}

func parsePage(filePath string, toFile bool) []Page {
	r, _ := core.Open(filePath)
	np := r.NumPage()
	doc := make([]Page, np)
	var wg sync.WaitGroup
	for i := 1; i <= np; i++ {
		wg.Add(1)
		go func(i int) {
			doc[i-1] = *NewPage(r.Page(i))
			if toFile {
				saveLinePng(&doc[i-1], filePath, i)
			}
			wg.Done()
		}(i)
	}
	wg.Wait()
	return doc
}

func saveLinePng(page *Page, filePath string, pageIdx int) {
	plt, _ := plot.New()
	plt.Add(plotter.NewGrid())
	plt.X.Min = page.MediaBox.Min.X
	plt.Y.Min = page.MediaBox.Min.Y
	plt.X.Max = page.MediaBox.Max.X
	plt.Y.Max = page.MediaBox.Max.Y
	for _, t := range page.Contents.Table {
		for _, c := range t.Cell {
			draw(plt, [4]float64{c.Min.X, c.Max.X, c.Max.Y, c.Max.Y}, 0, 0)
			draw(plt, [4]float64{c.Max.X, c.Max.X, c.Min.Y, c.Max.Y}, 0, 0)
			draw(plt, [4]float64{c.Min.X, c.Max.X, c.Min.Y, c.Min.Y}, 0, 0)
			draw(plt, [4]float64{c.Min.X, c.Min.X, c.Min.Y, c.Max.Y}, 0, 0)
		}
	}
	for _, l := range page.Contents.Line {
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

// Parse receives an array of paths and return array of content objects as JSON.
func Parse(filePaths []string, toFile bool) []byte {
	sTime := time.Now()
	docs := make([][]Page, len(filePaths))
	var wg sync.WaitGroup
	for i, fn := range filePaths {
		wg.Add(1)
		go func(fn string, i int) {
			docs[i] = parsePage(fn, toFile)
			wg.Done()
		}(fn, i)
	}
	wg.Wait()
	js, _ := json.MarshalIndent(docs, "", "  ")
	var result []byte
	if toFile {
		ioutil.WriteFile("pdf_parsed.json", js, os.ModePerm)
		eTime := time.Now()
		time := eTime.Sub(sTime).Seconds()
		result = []byte(fmt.Sprintf("complete. time: %fs\n", time))
	} else {
		result = js
	}
	return result
}
