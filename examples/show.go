// The show program demonstrates, with the help of gnuplot what the
// functions in the linear package can do.
package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"

	"zappem.net/pub/math/linear"
)

var (
	delta = flag.Float64("delta", 0.5, "displacement for basic box corners")
	angle = flag.Float64("angle", 5.0, "rotational angle counter clockwise")
	dX    = flag.Float64("dx", -2.0, "x-offset for box coordinates")
	dY    = flag.Float64("dy", 6.0, "y-offset for box coordinates")
)

func render(groups ...[]linear.Point) {
	fmt.Println("# X Y")
	for i, pts := range groups {
		for _, pt := range pts {
			fmt.Printf("%.3f %.3f\n", pt.X, pt.Y)
		}
		if i != len(groups)-1 {
			fmt.Println()
		}
	}
	fmt.Println("e")
}

func main() {
	box := []linear.Point{{1, 1}, {5, 1}, {5, 4}, {1, 4}}
	box2 := []linear.Point{{1, 1}, {5, 1}, {5 - *delta, 4}, {1 + *delta, 4}}
	c, s := math.Cos(*angle*math.Pi/180), math.Sin(*angle*math.Pi/180)
	zero := box2[0]
	for i, pt := range box2 {
		rX, rY := pt.X-zero.X, pt.Y-zero.Y
		box2[i].X = *dX + rX*c - rY*s
		box2[i].Y = *dY + rX*s + rY*c
	}

	aff, err := linear.DeriveAffine(box, box2)
	if err != nil {
		log.Fatalf("unable to generate affine from boxes: %v", err)
	}
	box = append(box, box[0])
	box2 = append(box2, box2[0])

	f := func(x float64) float64 {
		return (x-2)*(x-3)*(x-4) + rand.Float64()**delta
	}
	var pts, pts2 []linear.Point
	y0 := 1.0 - f(1.5)
	for x := 1.5; x < 4.5; x += 0.2 {
		pt := linear.Point{X: x, Y: f(x) + y0}
		pts = append(pts, pt)
		x2, y2 := aff.Apply(pt.X, pt.Y)
		pts2 = append(pts2, linear.Point{X: x2, Y: y2})
	}

	coeffs, err := linear.FitPoly(3, pts)
	if err != nil {
		log.Fatalf("failed to fit cubic poly: %v", err)
	}
	var fit, fit2 []linear.Point
	for _, pt := range pts {
		p := linear.Point{X: pt.X, Y: coeffs.Expand(pt.X)}
		fit = append(fit, p)
		fX, fY := aff.Apply(p.X, p.Y)
		fit2 = append(fit2, linear.Point{X: fX, Y: fY})
	}

	fmt.Println("set offsets 0.5, 0.5, 0.5, 0.5")
	fmt.Println()
	fmt.Println("plot '-' with lines title 'box', \\")
	fmt.Println("     '-' with lines title 'transformed', \\")
	fmt.Println("     '-' with points pt 1 ps 2 title 'data', \\")
	fmt.Println("     '-' with lines title 'fit'")

	render(box)
	render(box2)
	render(pts, pts2)
	render(fit, fit2)
}
