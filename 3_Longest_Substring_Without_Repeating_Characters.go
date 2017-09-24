package main

import "fmt"

func main() {
	fmt.Println(lengthOfLongestSubstring("你好你"))
}

/*
Given a string, find the length of the longest substring without repeating characters.
Examples:
Given "abcabcbb", the answer is "abc", which the length is 3.
Given "bbbbb", the answer is "b", with the length of 1.
Given "pwwkew", the answer is "wke", with the length of 3. Note that the answer must be a substring, "pwke" is a subsequence and not a substring.
*/

func lengthOfLongestSubstring(s string) int {
	runeStr := []rune(s) // Support Chinese
	strLen := len(runeStr)
	if strLen < 2 {
		return strLen
	}
	max := 0
	charMap := map[rune]int{}
	start := 0
	var tmpMax int
	for i, char := range runeStr {
		tmp, exists := charMap[char]
		charMap[char] = i
		if exists && tmp >= start {
			tmpMax = i - start
			if tmpMax > max {
				max = tmpMax
			}
			start = tmp + 1
		}
	}
	tmpMax = strLen - start
	if tmpMax > max {
		max = tmpMax
	}
	return max
}
