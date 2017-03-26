package main

import (
	"bytes"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

func main() {
	// fmt.Println(myAtoi("2147483648"))
	a := "23"
	fmt.Print(letterCombinations(a))
}

/* 1 */

// twoSum 给定整数数组，和目标和，在数组中找出和为目标和的两个数的下标
func twoSum(nums []int, target int) []int {
	numsMap := map[int]int{}
	for i, value := range nums {
		numsMap[value] = i
	}
	for i, value := range nums {
		delta := target - value
		index, exists := numsMap[delta]
		if exists && i != index {
			return []int{i, index}
		}
	}
	return nil
}

/* 2 */

// ListNode 链表节点定义
type ListNode struct {
	Val  int
	Next *ListNode
}

// addTwoNumbers 两个整数链表，相应节点相加，如果大于10则进位
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	var pResult *ListNode
	pResult = nil
	cur1 := l1
	cur2 := l2
	carry := 0
	var prevCur *ListNode
	prevCur = nil
	for cur1 != nil || cur2 != nil || carry != 0 {
		tmp := 0
		if cur1 != nil {
			tmp += cur1.Val
			cur1 = cur1.Next
		}
		if cur2 != nil {
			tmp += cur2.Val
			cur2 = cur2.Next
		}
		tmp += carry
		carry = tmp / 10
		tmp %= 10
		tmpNode := ListNode{tmp, nil}
		if pResult == nil {
			pResult = &tmpNode
		} else {
			(*prevCur).Next = &tmpNode
		}
		prevCur = &tmpNode
	}
	return pResult
}

/* 3 */

// lengthOfLongestSubstring 获取字符串中，字符不重复的最长子字符串的长度
func lengthOfLongestSubstring(s string) int {
	length := len(s)
	if length < 2 {
		return length
	}
	result := 0
	posTable := [256]int{}
	for index := range posTable {
		posTable[index] = -1
	}
	posStart := 0
	var i int
	for i = 0; i < length; i++ {
		tmp := posTable[s[i]]
		posTable[s[i]] = i
		if tmp >= posStart {
			if i-posStart > result {
				result = i - posStart
			}
			posStart = tmp + 1
		}
	}
	if i-posStart > result {
		result = i - posStart
	}
	return result
}

/* 4 */

// findKth 找出两个数组合并后第K小的数
func findKth(a []int, aStart int, b []int, bStart int, k int) int {
	lenA := len(a)
	lenB := len(b)
	if aStart >= lenA {
		return b[bStart+k-1]
	}
	if bStart >= lenB {
		return a[aStart+k-1]
	}
	if k == 1 {
		if a[aStart] < b[bStart] {
			return a[aStart]
		}
		return b[bStart]
	}
	intMax := int(^uint(0) >> 1)
	aKey := intMax
	bKey := intMax
	if aStart+k/2-1 < lenA {
		aKey = a[aStart+k/2-1]
	}
	if bStart+k/2-1 < lenB {
		bKey = b[bStart+k/2-1]
	}
	if aKey < bKey {
		return findKth(a, aStart+k/2, b, bStart, k-k/2)
	}
	return findKth(a, aStart, b, bStart+k/2, k-k/2)
}

// findMedianSortedArrays 计算两个有序数组合并后的中位数，要求算法复杂度为O(log(m+n))
func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
	len := len(nums1) + len(nums2)
	if len%2 == 1 {
		return float64(findKth(nums1, 0, nums2, 0, len/2+1))
	}
	return float64(findKth(nums1, 0, nums2, 0, len/2)+findKth(nums1, 0, nums2, 0, len/2+1)) / 2.0
}

func findMedianSortedArrays2(a []int, b []int) float64 {
	m, n := len(a), len(b)
	if m > n {
		a, b, m, n = b, a, n, m
	}
	if n == 0 {
		return 0
	}
	imin, imax, halfLen := 0, m, (m+n+1)/2
	var i, j, maxOfLeft, minOfRight int
	for imin <= imax {
		i = (imin + imax) / 2
		j = halfLen - i
		if i < m && b[j-1] > a[i] {
			imin = i + 1
		} else if i > 0 && a[i-1] > b[j] {
			imax = i - 1
		} else {
			if i == 0 {
				maxOfLeft = b[j-1]
			} else if j == 0 {
				maxOfLeft = a[i-1]
			} else {
				if a[i-1] > b[j-1] {
					maxOfLeft = a[i-1]
				} else {
					maxOfLeft = b[j-1]
				}
			}
			if (m+n)%2 == 1 {
				return float64(maxOfLeft)
			}
			if i == m {
				minOfRight = b[j]
			} else if j == n {
				minOfRight = a[i]
			} else {
				if a[i] < b[j] {
					minOfRight = a[i]
				} else {
					minOfRight = b[j]
				}
			}
			return float64(maxOfLeft+minOfRight) / 2.0
		}
	}
	return 0
}

/* 5 */
// longestPalindrome 找出字符串中的最长回文子字符串
func longestPalindrome(s string) string {
	length := len(s)
	if length == 0 {
		return ""
	}
	if length == 1 {
		return s
	}
	leftIndex, maxLen, left, right := 0, 1, 0, 0
	for start := 0; start < length && length-start > maxLen/2; {
		left, right = start, start
		for right < length-1 && s[right+1] == s[right] {
			right++
		}
		start = right + 1
		for right < length-1 && left > 0 && s[right+1] == s[left-1] {
			right++
			left--
		}
		if maxLen < right-left+1 {
			leftIndex = left
			maxLen = right - left + 1
		}
	}
	return string([]byte(s)[leftIndex : leftIndex+maxLen])
}

/* 6 */

// convert ZigZag格式输出字符串
func convert(s string, numRows int) string {
	length := len(s)
	if length == 0 || numRows <= 0 {
		return ""
	}
	if numRows == 1 {
		return s
	}
	var tempSlice = make([]byte, length)
	curr := 0
	for i := 0; i < numRows; i++ {
		for j := i; j < length; j += 2 * (numRows - 1) {
			tempSlice[curr] = s[j]
			curr++
			if i != 0 && i != numRows-1 {
				index := j + 2*(numRows-1) - 2*i
				if index < length {
					tempSlice[curr] = s[index]
					curr++
				}
			}
		}
	}
	return string(tempSlice)
}

/* 7 */
// reverse 整数翻转
func reverse(x int) int {
	if x == 0 {
		return x
	}
	sign := 1
	if x < 0 {
		sign = -1
		x = -x
	}
	if x < 10 {
		return sign * x
	}
	result := 0
	for x > 0 {
		result = 10*result + x%10
		x /= 10
	}
	result *= sign
	if result > math.MaxInt32 || result < math.MinInt32 {
		return 0
	}
	return result
}

/* 8 */
func isDigits(char byte) bool {
	return char >= '0' && char <= '9'
}

// myAtoi 字符串转整型
func myAtoi(str string) int {
	str = strings.Trim(str, " ")
	length := len(str)
	if length == 0 {
		return 0
	}
	sign, sum, curr := 1, 0, 0
	if str[0] == '+' {
		sign = 1
		curr++
	} else if str[0] == '-' {
		sign = -1
		curr++
	}
	for ; curr < length; curr++ {
		if !isDigits(str[curr]) {
			return sum * sign
		}
		sum = sum*10 + int(str[curr]-'0')
		if sign == 1 && sum > math.MaxInt32 {
			return math.MaxInt32
		}
		if sign == -1 && -sum < math.MinInt32 {
			return math.MinInt32
		}
	}
	return sum * sign
}
func strToInt32(s string) int {
	s = strings.Trim(s, " ")
	a, _ := strconv.Atoi(s)
	return a
}

/* 9 */
// isPalindrome 整数回文判断
func isPalindrome(x int) bool {
	if x < 0 {
		return false
	}
	if x < 10 {
		return true
	}
	count := 0
	tmp := x
	for tmp > 0 {
		tmp /= 10
		count++
	}
	i := 0
	j := count - 1
	tmp = int(math.Pow10(j))
	left := x
	right := x
	for i <= j {
		if left/tmp != right%10 {
			return false
		}
		left %= tmp
		right /= 10
		i++
		j--
		tmp /= 10
	}
	return true
}

/* 10 */
// isMatch 字符串对应
func isMatch(s string, p string) bool {
	lenS, lenP := len(s), len(p)
	dp := make([][]bool, lenS+1)
	for i := range dp {
		dp[i] = make([]bool, lenP+1)
	}
	dp[0][0] = true
	for i := 0; i <= lenS; i++ {
		for j := 1; j <= lenP; j++ {
			if p[j-1] == '*' {
				dp[i][j] = dp[i][j-2] || (i > 0 && (s[i-1] == p[j-2] || p[j-2] == '.') && dp[i-1][j])
			} else {
				dp[i][j] = i > 0 && dp[i-1][j-1] && (s[i-1] == p[j-1] || p[j-1] == '.')
			}
		}
	}
	return dp[lenS][lenP]
}

/* 11 */
// maxArea 蓄水池问题
func maxArea(height []int) int {
	length := len(height)
	if length < 2 {
		return 0
	}
	i, j, max, tmp := 0, length-1, 0, 0
	var heightI, heightJ int
	for i < j {
		heightI, heightJ = height[i], height[j]
		if height[i] < height[j] {
			tmp = (j - i) * height[i]
		} else {
			tmp = (j - i) * height[j]
		}
		if max < tmp {
			max = tmp
		}
		if height[i] < height[j] {
			for i < j && height[i] <= heightI {
				i++
			}
		} else {
			for i < j && height[j] <= heightJ {
				j--
			}
		}
	}
	return max
}

/* 12 */
// intToRoman 整数转罗马数，范围0~3999
func intToRoman(num int) string {
	M := []string{"", "M", "MM", "MMM"}
	C := []string{"", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"}
	X := []string{"", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"}
	I := []string{"", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"}
	/*
		buffer := bytes.Buffer{}
		buffer.WriteString(M[num/1000])
		buffer.WriteString(C[(num%1000)/100])
		buffer.WriteString(X[(num%100)/10])
		buffer.WriteString(I[num%10])
		return buffer.String()
	*/
	return M[num/1000] + C[(num%1000)/100] + X[(num%100)/10] + I[num%10]

}

/* 13 */
// romanToInt 罗马数字转整型 范围0~3999
func romanToInt(s string) int {
	length := len(s)
	nums := make([]int, length)
	for i, char := range s {
		switch char {
		case 'M':
			nums[i] = 1000
			break
		case 'D':
			nums[i] = 500
			break
		case 'C':
			nums[i] = 100
			break
		case 'L':
			nums[i] = 50
			break
		case 'X':
			nums[i] = 10
			break
		case 'V':
			nums[i] = 5
			break
		case 'I':
			nums[i] = 1
			break
		}
	}
	sum := 0
	for i := 0; i < length-1; i++ {
		if nums[i] < nums[i+1] {
			sum -= nums[i]
		} else {
			sum += nums[i]
		}
	}
	return sum + nums[length-1]
}

/* 14 */
// longestCommonPrefix 获取字符串数组的最长公共前缀字符串
func longestCommonPrefix(strs []string) string {
	count := len(strs)
	if count == 0 {
		return ""
	}
	if count == 1 {
		return strs[0]
	}
	prefixBuffer := bytes.Buffer{}
	lengthFirst := len(strs[0])
	for i := 0; i < lengthFirst; i++ {
		for j := 1; j < count; j++ {
			lengthPrev := len(strs[j-1])
			lengthCurr := len(strs[j])
			if i >= lengthCurr || i >= lengthPrev || strs[j-1][i] != strs[j][i] {
				return prefixBuffer.String()
			}
		}
		prefixBuffer.WriteByte(strs[0][i])
	}
	return prefixBuffer.String()
}

/* 15 */
// threeSum 给定整数数组，如果数组内存在三个整数和为0，那么将所有这样的三个数成组返回
func threeSum(nums []int) [][]int {
	length := len(nums)
	if length < 3 {
		return [][]int{}
	}
	sort.Ints(nums)
	result := make([][]int, 0)
	for i := 0; i < length-2; i++ {
		for i != 0 && i < length-2 && nums[i-1] == nums[i] {
			i++
		}
		target, left, right := -nums[i], i+1, length-1
		for left < right {
			sum := nums[left] + nums[right]
			if sum < target {
				for left < right && nums[left] == nums[left+1] {
					left++
				}
				left++
			} else if sum > target {
				for left < right && nums[right] == nums[right-1] {
					right--
				}
				right--
			} else {
				result = append(result, []int{nums[i], nums[left], nums[right]})
				for left < right && nums[left] == nums[left+1] {
					left++
				}
				for left < right && nums[right] == nums[right-1] {
					right--
				}
				left++
				right--
			}
		}
	}
	return result
}

/* 16 */
func myAbs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// threeSumClosest
func threeSumClosest(nums []int, target int) int {
	length := len(nums)
	if length <= 3 {
		sum := 0
		for _, value := range nums {
			sum += value
		}
		return sum
	}
	sort.Ints(nums)
	ans := nums[0] + nums[1] + nums[2]
	for i := 0; i < length-2; i++ {
		left := i + 1
		right := length - 1
		for left < right {
			sum := nums[i] + nums[left] + nums[right]
			if myAbs(ans-target) > myAbs(sum-target) {
				ans = sum
				if ans == target {
					return ans
				}
			}
			if sum > target {
				right--
			} else {
				left++
			}
		}
	}
	return ans
}

/* 17 */
// letterCombinations
func letterCombinations(digits string) []string {
	length := len(digits)
	if length == 0 {
		return []string{}
	}
	table := []string{
		0: "",
		1: "",
		2: "abc",
		3: "def",
		4: "ghi",
		5: "jkl",
		6: "mno",
		7: "pqrs",
		8: "tuv",
		9: "wxyz",
	}
	count := 1
	for _, num := range digits {
		index := num - '0'
		if index < 0 || index > 9 {
			break
		}
		if index == 0 || index == 1 {
			continue
		}
		count *= len(table[index])
	}
	result := make([]string, count)
	i := 0
	for _, num := range digits {
		index := num - '0'
		if index < 0 || index > 9 {
			break
		}
		if index == 0 || index == 1 {
			continue
		}
		i++
		for j, char := range table[index] {
			for k := range result {
				// TODO
				if j == count/len(table[index]) {
					result[k] += string(char)
				}
			}
		}
	}
	return result
}
