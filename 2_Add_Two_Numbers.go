package main

import (
	"fmt"
)

func main() {
	fmt.Println("你好啊")
}

/*
You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.
You may assume the two numbers do not contain any leading zero, except the number 0 itself.
Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
*/

// ListNode 链表节点定义
type ListNode struct {
	Val  int
	Next *ListNode
}

func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	pResult := &ListNode{}
	result := pResult
	pCurr1 := l1
	pCurr2 := l2
	carry := 0
	for pCurr1 != nil || pCurr2 != nil || carry != 0 {
		tmp := carry
		if pCurr1 != nil {
			tmp += pCurr1.Val
			pCurr1 = pCurr1.Next
		}
		if pCurr2 != nil {
			tmp += pCurr2.Val
			pCurr2 = pCurr2.Next
		}
		pResult.Next = &ListNode{tmp % 10, nil}
		carry = tmp / 10
		pResult = pResult.Next
	}
	return result.Next
}
