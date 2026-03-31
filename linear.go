// Package linear provides functions for numerical evaluation of
// selected methods in linear algebra.
package linear

import (
	"errors"
	"fmt"
	"math"
	"strings"
)

type Matrix [][]float64
type Coefs []float64

var (
	Zeroish               = 1e-12
	ErrNotSquare          = errors.New("not square")
	ErrNoInverse          = errors.New("no inverse")
	ErrInsufficientPoints = errors.New("insufficient points")
)

// String renders a matrix as multiple lines.
func (m Matrix) String() string {
	var rows []string
	for _, r := range m {
		var cols []string
		for _, c := range r {
			cols = append(cols, fmt.Sprintf(" %8.3g", c))
		}
		rows = append(rows, fmt.Sprint(cols))
	}
	return fmt.Sprintf("%s", strings.Join(rows, "\n"))
}

// Same confirms that two matrices are the same to a tolerance of
// Zeroish.
func (m Matrix) Same(n Matrix) bool {
	if len(m) != len(n) || len(m[0]) != len(n[0]) {
		return false
	}
	for i, mr := range m {
		nr := n[i]
		for j, a := range mr {
			if math.Abs(a-nr[j]) > Zeroish {
				return false
			}
		}
	}
	return true
}

// XM multiplies a matrix by a matrix, returning the product matrix.
func (m Matrix) XM(n Matrix) Matrix {
	if len(m[0]) != len(n) {
		panic("invalid matrix matrix product")
	}
	u := make(Matrix, len(m))
	for i := 0; i < len(m); i++ {
		for j := 0; j < len(n[0]); j++ {
			val := 0.0
			for k, r := range m[i] {
				val += r * n[k][j]
			}
			u[i] = append(u[i], val)
		}
	}
	return u
}

// SubRows subtracts alpha*row1 from row2, and multiplies row1 by norm.
// It modifies the matrix in place.
func (m Matrix) SubRows(row1, row2 int, norm, alpha float64) {
	r := m[row2]
	for i, a := range m[row1] {
		b := r[i]
		m[row2][i] = b - alpha*a
		m[row1][i] = a * norm
	}
}

// CombineRows changes the rows of a matrix by summing the rows with
// coefficients alpha and beta to replace row1 and replaces row2 with
// the difference of the same row combinations. It modifies the matrix
// in place.
func (m Matrix) CombineRows(row1, row2 int, alpha, beta float64) {
	r1, r2 := m[row1], m[row2]
	for i, v := range r1 {
		u := r2[i]
		r1[i] = alpha*v + beta*u
		r2[i] = alpha*v - beta*u
	}
}

// Dissolve forces values close to Zeroish to be 0 to clean up a
// matrix. It modifies the matrix in place.
func (m Matrix) Dissolve() {
	for _, v := range m {
		for i, x := range v {
			if math.Abs(x) <= Zeroish {
				v[i] = 0
			}
		}
	}
}

// Duplicate returns a full copy of a matrix
func (m Matrix) Duplicate() Matrix {
	var n Matrix
	for _, v := range m {
		u := make([]float64, len(v))
		copy(u, v)
		n = append(n, u)
	}
	return n
}

// Unit generates the unit matrix of order width.
func Unit(n int) Matrix {
	var m Matrix
	for i := 0; i < n; i++ {
		v := make([]float64, n)
		v[i] = 1
		m = append(m, v)
	}
	return m
}

// Inv returns the inverted matrix. It only operates on a square
// matrix.
func (m Matrix) Inv() (Matrix, error) {
	if len(m) != len(m[0]) {
		return nil, ErrNotSquare
	}
	a := m.Duplicate()
	b := Unit(len(a))
	for c := 0; c < len(a); c++ {
		for r := c + 1; r < len(a); r++ {
			x, y := a[c][c], a[r][c]
			if math.Abs(x+y) <= Zeroish {
				if x > y {
					a.CombineRows(c, r, 1, -1)
					b.CombineRows(c, r, 1, -1)
				} else {
					a.CombineRows(c, r, -1, 1)
					b.CombineRows(c, r, -1, 1)
				}
			} else if math.Abs(x) <= Zeroish {
				a.SubRows(r, c, 1, 1)
				b.SubRows(r, c, 1, 1)
				a.SubRows(c, r, 1, -1)
				b.SubRows(c, r, 1, -1)
			} else {
				a.CombineRows(c, r, y/(x+y), x/(x+y))
				b.CombineRows(c, r, y/(x+y), x/(x+y))
			}
		}
	}
	for c := len(a); c > 0; {
		c--
		for r := c; r > 0; {
			r--
			x, y := a[c][c], a[r][c]
			a.SubRows(c, r, 1/x, y/x)
			b.SubRows(c, r, 1/x, y/x)
		}
	}
	b.SubRows(0, 1, 1/a[0][0], 0)
	b.Dissolve()
	return b, nil
}

// Point is a two dimensional coordinate.
type Point struct {
	X, Y float64
}

// FitPoly takes a series of xy points and derives close fit
// coefficients for a polynomial up to x^n. The function errors out if
// the number of points supplied is less than n+1.
func FitPoly(n int, xy []Point) (Coefs, error) {
	if n+1 > len(xy) {
		return nil, ErrInsufficientPoints
	}
	var norm float64
	for _, coord := range xy {
		if x := math.Abs(coord.X); norm < x {
			norm = x
		}
	}
	if norm == 0.0 {
		return make([]float64, n+1), nil
	}

	xnormal := .5 / norm
	d := make([]float64, 2*n+2)
	e := make([]float64, n+1)
	value := 1.0
	for _, coord := range xy {
		for i := 0; i <= 2*n; i++ {
			if i <= n {
				e[i] += coord.Y * value
			}
			d[i] += value
			value *= xnormal * coord.X
		}
	}
	var a Matrix
	var b Matrix
	for i := 0; i <= n; i++ {
		b = append(b, []float64{e[i]})
		var row []float64
		for j := 0; j <= n; j++ {
			row = append(row, d[i+j])
		}
		a = append(a, row)
	}
	recip, err := a.Inv()
	if err != nil {
		return nil, err
	}
	soln := recip.XM(b)
	var coefs Coefs
	xf := 1.0
	for _, v := range soln {
		coefs = append(coefs, v[0]*xf)
		xf *= xnormal
	}
	return coefs, nil
}

// Expand performs the series expansion of x with cs coefficients.
func (cs Coefs) Expand(x float64) float64 {
	q := 1.0
	y := 0.0
	for _, c := range cs {
		y += q * c
		q *= x
	}
	return y
}
