## Python 中使用 logging 模块代替 print

**使用logging模块**

logging模块是Python内置的日志模块，使用它可以非常轻松的处理和管理日志输出。 logging模块最简单的用法，是直接使用basicConfig方法来对logging进行配置：

```python
import logging
# 设置默认的level为DEBUG
# 设置log的格式
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(name)s:%(levelname)s: %(message)s"
)

# 记录log
logging.debug(...)
logging.info(...)
logging.warning(...)
logging.error(...)
logging.critical(...)
```

这样配置完logging以后，然后使用``logging.debug``来替换所有的print语句就可以了。 我们会看到这样的输出：

```
[2014-03-18 15:17:45,216] root:cur: 0, start: 1, end: 100
[2014-03-18 15:17:45,216] root:DEBUG: cur: 1, start: 1, end: 100
[2014-03-18 15:17:45,216] root:DEBUG: Returning result 1
[2014-03-18 15:17:45,216] root:DEBUG: Before caculation: a, b = 0, 1
```

**使用真正的logger**

上面说的basicConfig方法可以满足你在绝大多数场景下的使用需求，但是basicConfig有一个 很大的缺点。

调用basicConfig其实是给root logger添加了一个handler，这样当你的程序和别的使用了 logging的第三方模块一起工作时，会影响第三方模块的logger行为。这是由logger的继承特性决定的。

所以我们需要使用真正的logger：

```python
import logging
# 使用一个名字为fib的logger
logger = logging.getLogger('fib')

# 设置logger的level为DEBUG
logger.setLevel(logging.DEBUG)

# 创建一个输出日志到控制台的StreamHandler
hdr = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(name)s:%(levelname)s: %(message)s')
hdr.setFormatter(formatter)

# 给logger添加上handler
logger.addHandler(hdr)
```

**动态控制脚本的所有输出**

使用了logging模块以后，通过修改logger的log level，我们就可以方便的控制程序的输出了。 比如我们可以为我们的斐波那契数列添加一个 -v 参数，来控制打印所有的调试信息。

```python
# 添加接收一个verbose参数
parser.add_argument('-v', '--verbose', action='store_true', dest='verbose',
          help='Enable debug info')



# 判断verbose
if args.verbose:
  logger.setLevel(logging.DEBUG)
else:
  logger.setLevel(logging.ERROR)
```

这样，默认情况下，我们的小程序是不会打印调试信息的，只有当传入`-v/--verbose`的时候， 我们才会打印出额外的debug信息，就像这样：

```
$ python fib.py -s 1 -e 100
1 1 2 3 5 8 13 21 34 55 89



$ python fib.py -s 1 -e 100 -v
[2014-03-18 15:17:45,216] fib:DEBUG: cur: 0, start: 1, end: 100
[2014-03-18 15:17:45,216] fib:DEBUG: cur: 1, start: 1, end: 100
[2014-03-18 15:17:45,216] fib:DEBUG: Returning result 1
[2014-03-18 15:17:45,216] fib:DEBUG: Before caculation: a, b = 0, 1
... ...
```

使用了logging以后，什么时候需要打印DEBUG信息，什么时候需要关闭， 一切变的无比简单。





















































