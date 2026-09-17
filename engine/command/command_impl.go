package command

type command struct {
	cmdType CommandType
	cb      CommandFunc
}
