package main

import (
	"fmt"
)

func main() {
	a := []int{1, 2, 3}
	fmt.Println(twoSum(a, 3))
}

/*
 *Given an array of integers, return indices of the two numbers such that they add up to a specific target.
 *You may assume that each input would have exactly one solution, and you may not use the same element twice.
 */

func twoSum(nums []int, target int) []int {
	numsMap := make(map[int]int, len(nums))
	for i, value := range nums {
		index, exists := numsMap[target-value]
		if exists {
			return []int{index, i}
		}
		numsMap[value] = i
	}
	return nil
}
